"""Evaluate an existing results pkl against the same nuScenes split."""

import argparse
import importlib
import os
import sys

import mmcv
from mmcv import Config

sys.path.append(".")

from mmdet3d.datasets import build_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="projects/configs/petr/petr_vovnet_gridmask_p4_800x320_export.py")
    p.add_argument("--data_root", default="data/nuscenes/")
    p.add_argument("--ann_file", default="data/nuscenes/nuscenes_infos_val.pkl")
    p.add_argument("--pkl", required=True)
    p.add_argument("--eval", nargs="+", default=["bbox"])
    return p.parse_args()


def import_plugin(cfg, config_path):
    if not getattr(cfg, "plugin", False):
        return
    plugin_dir = getattr(cfg, "plugin_dir", None) or os.path.dirname(config_path)
    importlib.import_module(".".join(plugin_dir.split("/")))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg, args.config)
    cfg.data.test.data_root = args.data_root
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.pkl)
    eval_kwargs = cfg.get("evaluation", {}).copy()
    for k in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "pipeline"]:
        eval_kwargs.pop(k, None)
    eval_kwargs["metric"] = args.eval
    print(dataset.evaluate(results, **eval_kwargs))


if __name__ == "__main__":
    main()
