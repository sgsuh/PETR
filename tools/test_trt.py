"""
Create: 2026.05.09
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0

TensorRT inference over the nuScenes mini dataset with optional AB3DMOT
tracking, followed by the standard mmdet3d nuScenes evaluation.
"""

import argparse
import importlib
import os
import sys
import warnings

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction

sys.path.append(".")
sys.path.append("./tools/")

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet.core.bbox.builder import BBOX_CODERS

from singleshot.trt import TRT


def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT eval on nuScenes mini")

    parser.add_argument("--config",
                        default="projects/configs/petr/petr_vovnet_gridmask_p4_800x320_export.py")
    parser.add_argument("--onnx_path",
                        default="ckpts/petr-vov-p4-800x320/epoch_24.onnx")
    parser.add_argument("--trt_path",
                        default="ckpts/petr-vov-p4-800x320/epoch_24.engine")
    parser.add_argument("--data_root",
                        default="data/nuscenes/")
    parser.add_argument("--ann_file",
                        default="data/nuscenes/nuscenes_infos_val.pkl")
    parser.add_argument("--out",
                        default="work_dirs/petr_vov_trt/results_eval.pkl")
    parser.add_argument("--eval",
                        nargs="+",
                        default=["bbox"])
    parser.add_argument("--track",
                        action="store_true",
                        help="apply AB3DMOT tracker to the detections before evaluation")
    parser.add_argument("--match_distance",
                        default="iou",
                        choices=["iou", "m"])
    parser.add_argument("--match_threshold",
                        type=float,
                        default=0.1)
    parser.add_argument("--match_algorithm",
                        default="greedy")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--workers_per_gpu",
                        type=int,
                        default=0)
    parser.add_argument("--cfg-options",
                        nargs="+",
                        action=DictAction)
    parser.add_argument("--eval-options",
                        nargs="+",
                        action=DictAction)

    return parser.parse_args()


def import_plugin(cfg, config_path):
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if not getattr(cfg, "plugin", False):
        return

    plugin_dir = getattr(cfg, "plugin_dir", None) or os.path.dirname(config_path)
    module_path = ".".join(plugin_dir.split("/"))
    print(f"importing plugin: {module_path}")
    importlib.import_module(module_path)


def build_trackers(class_names):
    from tools.object_tracker.AD3DMOT import AB3DMOT

    return {name: AB3DMOT(covariance_id=0,
                          tracking_name=name,
                          use_angular_velocity=False,
                          tracking_nuscenes=True)
            for name in class_names}


def compute_img2lidars(img_metas):
    """Invert per-camera lidar2img matrices into img2lidar (B, N, 4, 4)."""
    per_sample = []
    for meta in img_metas:
        mats = [np.linalg.inv(np.asarray(m)) for m in meta["lidar2img"]]
        per_sample.append(np.stack(mats, axis=0))
    return torch.from_numpy(np.stack(per_sample, axis=0)).float()


def to_decoder_input(cls_scores, bbox_preds):
    """TRT engine returns the last decoder layer; the coder slices [-1]."""
    if cls_scores.dim() == 3:
        cls_scores = cls_scores.unsqueeze(0)
        bbox_preds = bbox_preds.unsqueeze(0)
    return {"all_cls_scores": cls_scores,
            "all_bbox_preds": bbox_preds,
            "enc_cls_scores": None,
            "enc_bbox_preds": None}


def apply_tracker(pred_dict, class_names, trackers, args):
    """AB3DMOT input row order: [h, w, l, x, y, z, yaw]; info: [score]."""
    bboxes = pred_dict["bboxes"].cpu().numpy()
    scores = pred_dict["scores"].cpu().numpy()
    labels = pred_dict["labels"].cpu().numpy().astype(int)

    dets_per = {n: [] for n in class_names}
    info_per = {n: [] for n in class_names}

    for i in range(len(labels)):
        name = class_names[labels[i]]
        cx, cy, cz, w, l, h, yaw = (bboxes[i, 0], bboxes[i, 1], bboxes[i, 2],
                                    bboxes[i, 3], bboxes[i, 4], bboxes[i, 5],
                                    bboxes[i, 6])
        dets_per[name].append([h, w, l, cx, cy, cz, yaw])
        info_per[name].append([scores[i]])

    out_bbox, out_score, out_label = [], [], []

    for label_id, name in enumerate(class_names):
        if not dets_per[name]:
            continue

        dets_all = {"dets": np.array(dets_per[name]),
                    "info": np.array(info_per[name])}
        result = trackers[name].update(dets_all,
                                       args.match_distance,
                                       args.match_threshold,
                                       args.match_algorithm)

        if result.shape[0] == 0:
            continue

        # result row layout: [h, w, l, x, y, z, yaw, track_id, track_score]
        for row in result:
            h_, w_, l_, x, y, z, yaw = row[:7]
            track_score = row[8] if row.shape[0] > 8 else row[-1]
            out_bbox.append([x, y, z, w_, l_, h_, yaw, 0.0, 0.0])
            out_score.append(track_score)
            out_label.append(label_id)

    if not out_bbox:
        return {"bboxes": torch.zeros((0, 9), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long)}

    return {"bboxes": torch.tensor(out_bbox, dtype=torch.float32),
            "scores": torch.tensor(out_score, dtype=torch.float32),
            "labels": torch.tensor(out_label, dtype=torch.long)}


def to_pts_bbox_result(pred_dict, box_type_3d):
    bboxes = pred_dict["bboxes"].cpu().clone()
    # NMSFreeCoder returns gravity-center z; LiDARInstance3DBoxes wants
    # bottom-center z (origin = (0.5, 0.5, 0)). Match get_bboxes() in
    # PETRHeadExport which subtracts half-height before wrapping.
    if bboxes.numel() > 0:
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

    box_obj = box_type_3d(bboxes, box_dim=bboxes.size(-1))
    return dict(boxes_3d=box_obj,
                scores_3d=pred_dict["scores"].cpu(),
                labels_3d=pred_dict["labels"].cpu().long())


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_plugin(cfg, args.config)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.data_root = args.data_root
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=args.batch_size,
                                   workers_per_gpu=args.workers_per_gpu,
                                   dist=False,
                                   shuffle=False)

    coder_cfg = cfg.model.pts_bbox_head.bbox_coder.copy()
    bbox_coder = BBOX_CODERS.build(coder_cfg)

    class_names = list(cfg.class_names)
    trackers = build_trackers(class_names) if args.track else None

    if not os.path.isfile(args.trt_path) and not os.path.isfile(args.onnx_path):
        raise FileNotFoundError(
            f"Neither TRT engine ({args.trt_path}) nor ONNX file ({args.onnx_path}) exists")

    trt_model = TRT(args)

    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))

    for data in data_loader:
        img = data["img"][0].data[0]
        img_metas = data["img_metas"][0].data[0]
        img2lidars = compute_img2lidars(img_metas)

        cls_scores, bbox_preds = trt_model.inference(img, img2lidars)
        preds = to_decoder_input(cls_scores, bbox_preds)
        decoded = bbox_coder.decode(preds)

        for b, pred in enumerate(decoded):
            if args.track:
                pred = apply_tracker(pred, class_names, trackers, args)
            results.append({"pts_bbox": to_pts_bbox_result(pred, img_metas[b]["box_type_3d"])})

        prog_bar.update()

    print()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        print(f"writing results to {args.out}")
        mmcv.dump(results, args.out)

    eval_kwargs = cfg.get("evaluation", {}).copy()

    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "pipeline"]:
        eval_kwargs.pop(key, None)

    eval_kwargs.update(dict(metric=args.eval, **(args.eval_options or {})))
    print(dataset.evaluate(results, **eval_kwargs))


if __name__ == "__main__":
    main()
