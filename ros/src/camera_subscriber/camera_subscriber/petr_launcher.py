"""
Create: 2026.05.24
Author: SG.SUH
Python: 3.8.10
PyTorch: 1.14.0
"""

import importlib
import os
import sys

# `tools.*` and `projects.*` are top-level packages under /workspace/PETR,
# not installed via pip. `sys.path.append(".")` only works if the node is
# launched from /workspace/PETR, which is not the case under `ros2 run`.
# Pin the absolute repo root so the imports below resolve regardless of CWD.
PETR_ROOT = "/workspace/PETR"
if PETR_ROOT not in sys.path:
    sys.path.insert(0, PETR_ROOT)

import cv2
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

from tools.singleshot.petr_custom import petr_custom
from tools.singleshot.trt import TRT

class ARGS:
    def __init__(self):
        self.config = "/workspace/PETR/projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py"
        self.checkpoint = "/workspace/PETR/ckpts/petr-vov-p4-800x320/epoch_24.pth"
        self.out = "/workspace/PETR/work_dirs/pp-nus/results_eval.pkl"
        self.cfg_options = None
        self.deterministic = False
        self.eval = None
        self.eval_options = {"jsonfile_prefix": "/workspace/PETR/work_dirs/pp-nus/results_eval"}
        self.format_only = True
        self.fuse_conv_bin = False
        self.gpu_collect = False
        self.launcher = "none"
        self.local_rank = 0
        self.options = None
        self.seed = 0
        self.show = False
        self.show_dir = None
        self.tmpdir = None
        self.batch_size = 1
        self.onnx_path = "/workspace/PETR/ckpts/petr-vov-p4-800x320/epoch_24_v2.onnx"
        self.trt_path = "/workspace/PETR/ckpts/petr-vov-p4-800x320/epoch_24_v2.engine"

class PETR_class:
    def __init__(self):
        args = ARGS()

        assert args.out or args.eval or args.format_only or args.show or args.show_dir, ("Please specify at least one operation (save/eval/format/show the results / save the results) with the argument --out, --eval, --format-only, --show or --show-dir")

        if args.eval and args.format_only:
            raise ValueError("--eval and --format_only cannot be both specified")
        
        if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
            raise ValueError("The output file must be a pkl file.")
        
        self.cfg = Config.fromfile(args.config)
        importlib.import_module("projects.mmdet3d_plugin")
        self.cfg.model.pretrained = None
        self.cfg.model.train_cfg = None
        self.is_trt = os.path.isfile(args.onnx_path)

        if not self.is_trt:
            model = build_model(self.cfg.model, test_cfg=(self.cfg.get("test_cfg")))
            checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
            model.CLASSES = checkpoint["meta"]["CLASSES"]
            model.PALETTE = checkpoint["meta"]["PALETTE"]
            model = MMDataParallel(model, device_ids=[0])
            model.eval()
        else:
            model = TRT(args)

        self.petr = petr_custom(model, petrv2=False)

    def detect_object(self,
                      image):
        image_data = cv2.resize(image, (1600, 900))

        return self.petr.doPetr(self.cfg, image_data, 0.25, self.is_trt)
    
    def draw_picture(self,
                     image):
        return self.petr.just_draw_same_bbox(image)