"""
Create: 2026-05-23
Author: SG.SUH
Python: 3.8.13

Three-way numeric diff to localize the TRT metric regression:

  A. training-equivalent eager  : model.pts_bbox_head(feats, img_metas)
  B. ONNX-traced wrapper eager  : PETRExportWrapper._head_forward(...)
  C. TensorRT engine            : TRT.inference(img, img2lidars)

A vs B isolates the wrapper rewrite (mask/pad/img2lidar plumbing,
torch.maximum vs torch.clamp, missing torch.nan_to_num, expand vs repeat,
etc.). B vs C isolates ONNX/TRT numerics (atan2 surrogate, opset folding,
TRT kernel selection).

Run:
  python tools/compare_eager_trt.py \\
      --config projects/configs/petr/petr_vovnet_gridmask_p4_800x320_export.py \\
      --checkpoint ckpts/petr-vov-p4-800x320/epoch_24.pth \\
      --trt_path ckpts/petr-vov-p4-800x320/epoch_24_v2.engine \\
      --onnx_path ckpts/petr-vov-p4-800x320/epoch_24_v2.onnx \\
      --sample_index 0
"""

import argparse
import importlib
import os
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

sys.path.append(".")
sys.path.append("./tools/")

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset


BBOX_COMPONENTS = ["cx", "cy", "w", "l", "cz", "h", "sin", "cos", "vx", "vy"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="projects/configs/petr/petr_vovnet_gridmask_p4_800x320_export.py")
    p.add_argument("--checkpoint",
                   default="ckpts/petr-vov-p4-800x320/epoch_24.pth")
    p.add_argument("--data_root", default="data/nuscenes/")
    p.add_argument("--ann_file", default="data/nuscenes/nuscenes_infos_val.pkl")
    p.add_argument("--trt_path",
                   default="ckpts/petr-vov-p4-800x320/epoch_24_v2.engine")
    p.add_argument("--onnx_path",
                   default="ckpts/petr-vov-p4-800x320/epoch_24_v2.onnx")
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def import_plugin(cfg, config_path):
    if not getattr(cfg, "plugin", False):
        return
    plugin_dir = getattr(cfg, "plugin_dir", None) or os.path.dirname(config_path)
    importlib.import_module(".".join(plugin_dir.split("/")))


def report_diff(name, a, b, components=None):
    diff = (a - b).abs()
    print(f"  Δ{name:8s} max={diff.max().item():.4e}  "
          f"mean={diff.mean().item():.4e}  "
          f"median={diff.median().item():.4e}")
    if components is not None and a.shape[-1] == len(components):
        for k, n in enumerate(components):
            d = (a[..., k] - b[..., k]).abs()
            print(f"      [{n:>3s}] max={d.max().item():.4e}  "
                  f"mean={d.mean().item():.4e}")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg, args.config)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.data_root = args.data_root
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset,
                              samples_per_gpu=args.batch_size,
                              workers_per_gpu=0,
                              dist=False,
                              shuffle=False)

    data = None
    for i, d in enumerate(loader):
        if i == args.sample_index:
            data = d
            break
    assert data is not None, f"sample_index {args.sample_index} out of range"

    img = data["img"][0].data[0]
    img_metas = data["img_metas"][0].data[0]

    print(f"sample {args.sample_index} : img={tuple(img.shape)}  "
          f"img_shape={img_metas[0]['img_shape'][0]}  "
          f"pad_shape={img_metas[0]['pad_shape'][0]}")

    img2lidars_np = []
    for m in img_metas:
        mats = [np.linalg.inv(np.asarray(x)) for x in m["lidar2img"]]
        img2lidars_np.append(np.stack(mats, 0))
    img2lidars_cpu = torch.from_numpy(np.stack(img2lidars_np, 0)).float()
    img2lidars = img2lidars_cpu.cuda()

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.cuda().eval()

    # Path A needs the training detector + training head (Petr3D / PETRHead),
    # not the export variants whose forwards take different signatures.
    from copy import deepcopy
    cfg_train = deepcopy(cfg)
    cfg_train.model.type = "Petr3D"
    cfg_train.model.pts_bbox_head.type = "PETRHead"
    cfg_train.model.pts_bbox_head.pop("img_height", None)
    cfg_train.model.pts_bbox_head.pop("img_width", None)
    model_train = build_model(cfg_train.model, test_cfg=cfg_train.get("test_cfg"))
    load_checkpoint(model_train, args.checkpoint, map_location="cpu")
    model_train.cuda().eval()

    img_cuda = img.cuda()

    # ---- Path A : training-equivalent eager -------------------------------
    # Petr3D.extract_img_feat does img.squeeze_() in place, so feed a clone.
    with torch.no_grad():
        feats = model_train.extract_img_feat(img_cuda.clone(), img_metas)
        outs_a = model_train.pts_bbox_head(feats, img_metas)
    bbox_a = outs_a["all_bbox_preds"][-1].detach().cpu()
    cls_a  = outs_a["all_cls_scores"][-1].detach().cpu()
    print(f"[A eager-orig] bbox_preds shape={tuple(bbox_a.shape)}  "
          f"cls shape={tuple(cls_a.shape)}")

    # ---- Path B : wrapper (ONNX-traced code) eager ------------------------
    from export_onnx import PETRExportWrapper
    wrapper = PETRExportWrapper(model).cuda().eval()
    with torch.no_grad():
        feats_b = wrapper._extract_img_feats(img_cuda)
        H, W = img_cuda.shape[-2], img_cuda.shape[-1]
        cls_b_all, bbox_b_all = wrapper._head_forward(feats_b, img2lidars, H, W)
    bbox_b = bbox_b_all[-1].detach().cpu()
    cls_b  = cls_b_all[-1].detach().cpu()

    print()
    print("=== A vs B  (training-eager  vs  ONNX-wrapper-eager) ===")
    report_diff("bbox", bbox_a, bbox_b, components=BBOX_COMPONENTS)
    report_diff("cls",  cls_a,  cls_b)

    # ---- Path C : TRT engine ---------------------------------------------
    if not os.path.isfile(args.trt_path):
        print(f"\n[C] TRT engine {args.trt_path} missing — skipping TRT diff")
        return

    # End-to-end eager (wrapper.forward) for apples-to-apples with TRT outputs
    with torch.no_grad():
        bb_e, sc_e, lb_e = wrapper(img_cuda, img2lidars)

    class _A: pass
    a = _A()
    a.trt_path = args.trt_path
    a.onnx_path = args.onnx_path
    a.batch_size = args.batch_size

    from singleshot.trt import TRT
    trt_model = TRT(a)
    bb_t, sc_t, lb_t = trt_model.inference(img, img2lidars_cpu)

    bb_e, sc_e, lb_e = bb_e.cpu(), sc_e.cpu(), lb_e.cpu()
    bb_t, sc_t, lb_t = bb_t.cpu(), sc_t.cpu(), lb_t.cpu()

    print()
    print("=== B vs C  (wrapper-eager-decoded  vs  TRT-decoded) ===")
    # decoded format: [cx,cy,cz_bottom,w,l,h,yaw,vx,vy]
    DECODED = ["cx","cy","cz","w","l","h","yaw","vx","vy"]
    report_diff("bbox", bb_e, bb_t, components=DECODED)
    report_diff("scores", sc_e, sc_t)
    report_diff("labels", lb_e.float(), lb_t.float())

    # The position-wise diff above is misleading: each path's top-K sort can
    # shuffle ties differently. Re-pair by matching every eager box to its
    # nearest TRT box (by 3D center).
    e_xyz = bb_e[0, :, :3]
    t_xyz = bb_t[0, :, :3]
    d_mat = torch.cdist(e_xyz, t_xyz)               # (300, 300)
    nn_dist, nn_idx = d_mat.min(dim=1)              # for every eager box
    matched_yaw = bb_t[0, nn_idx, 6]
    matched_lab = lb_t[0, nn_idx]
    print()
    print("  -- nearest-neighbour match (eager → TRT) --")
    print(f"  match-distance      max={nn_dist.max().item():.4e}  "
          f"mean={nn_dist.mean().item():.4e}  "
          f"median={nn_dist.median().item():.4e}")
    same_lbl = (lb_e[0] == matched_lab).float().mean().item()
    print(f"  label agreement on matched pairs : {same_lbl*100:.1f}%")
    d_yaw_m = (bb_e[0, :, 6] - matched_yaw).abs()
    # wrap to [0, π]
    d_yaw_w = torch.minimum(d_yaw_m, (2 * np.pi - d_yaw_m).abs())
    print(f"  Δyaw (wrap-aware)   max={d_yaw_w.max().item():.4e}  "
          f"mean={d_yaw_w.mean().item():.4e}  "
          f"median={d_yaw_w.median().item():.4e}")

    # Sanity: does the atan2 surrogate, applied to the eager sin/cos, agree
    # with torch.atan2?  This is what the ONNX exporter substituted.
    sin = bbox_b[..., 6]
    cos = bbox_b[..., 7]
    truth = torch.atan2(sin, cos)
    eps = 1e-12
    surrogate = 2.0 * torch.atan(sin / (torch.sqrt(sin * sin + cos * cos) + cos + eps))
    d_surr = (truth - surrogate).abs()
    d_surr = torch.minimum(d_surr, (2 * np.pi - d_surr).abs())
    print()
    print(f"  atan2 surrogate self-check on eager sin/cos: "
          f"max={d_surr.max().item():.4e}  mean={d_surr.mean().item():.4e}")
    if d_surr.max().item() > 1e-3:
        worst = torch.topk(d_surr.flatten(), k=5)
        for v, i in zip(worst.values, worst.indices):
            print(f"    sin={sin.flatten()[i]:+.4e}  cos={cos.flatten()[i]:+.4e}  "
                  f"truth={truth.flatten()[i]:+.4f}  surr={surrogate.flatten()[i]:+.4f}")


if __name__ == "__main__":
    main()
