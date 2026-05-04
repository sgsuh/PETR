"""
Create: 2022.10.25
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import sys

sys.path.append(".")

import argparse
import os
import importlib

import torch
import torch.nn as nn
import onnx

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet3d.models import build_model
from mmdet.models.utils.transformer import inverse_sigmoid

from projects.mmdet3d_plugin.models.dense_heads.petr_head import pos2posemb3d


class PETRExportWrapper(nn.Module):
    """Wraps a Petr3D detector with an ONNX-friendly forward(img, img2lidars).

    The original Petr3D.forward dispatches by `return_loss` and the head reads
    image padding / lidar2img matrices from Python-level `img_metas`, neither
    of which can be traced. This wrapper inlines the inference path and
    consumes `img2lidars` as a plain tensor.
    """

    def __init__(self, petr_model):
        super().__init__()
        self.model = petr_model

    def _extract_img_feats(self, img):
        # img: (B, N, 3, H, W)
        B, N, C, H, W = img.shape
        img_flat = img.view(B * N, C, H, W)

        feats = self.model.img_backbone(img_flat)
        if isinstance(feats, dict):
            feats = list(feats.values())
        if self.model.with_img_neck:
            feats = self.model.img_neck(feats)

        reshaped = []
        for f in feats:
            _, c, h, w = f.size()
            reshaped.append(f.view(B, N, c, h, w))
        return reshaped

    def _position_embedding(self, img_feats, img2lidars, pad_h, pad_w):
        head = self.model.pts_bbox_head
        eps = 1e-5
        x = img_feats[head.position_level]
        B, N, _, H, W = x.shape
        device = x.device
        dtype = x.dtype

        coords_h = torch.arange(H, device=device, dtype=dtype) * pad_h / H
        coords_w = torch.arange(W, device=device, dtype=dtype) * pad_w / W

        if head.LID:
            index = torch.arange(head.depth_num, device=device, dtype=dtype)
            index_1 = index + 1
            bin_size = (head.position_range[3] - head.depth_start) / (
                head.depth_num * (1 + head.depth_num)
            )
            coords_d = head.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(head.depth_num, device=device, dtype=dtype)
            bin_size = (head.position_range[3] - head.depth_start) / head.depth_num
            coords_d = head.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(
            torch.meshgrid([coords_w, coords_h, coords_d])
        ).permute(1, 2, 3, 0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps
        )

        i2l = img2lidars.to(dtype).view(B, N, 1, 1, 1, 4, 4).expand(B, N, W, H, D, 4, 4)
        coords = coords.view(1, 1, W, H, D, 4, 1).expand(B, N, W, H, D, 4, 1)
        coords3d = torch.matmul(i2l, coords).squeeze(-1)[..., :3]

        pr = head.position_range
        coords3d[..., 0:1] = (coords3d[..., 0:1] - pr[0]) / (pr[3] - pr[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - pr[1]) / (pr[4] - pr[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - pr[2]) / (pr[5] - pr[2])

        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_pe = head.position_encoder(coords3d)
        return coords_pe.view(B, N, head.embed_dims, H, W)

    def _head_forward(self, mlvl_feats, img2lidars, pad_h, pad_w):
        head = self.model.pts_bbox_head
        x = mlvl_feats[0]
        B, N = x.size(0), x.size(1)

        # Assume no padding in the exported graph: full image is valid, masks all-False.
        x = head.input_proj(x.flatten(0, 1))
        x = x.view(B, N, *x.shape[-3:])
        masks = x.new_zeros((B, N, x.shape[-2], x.shape[-1]), dtype=torch.bool)

        if head.with_position:
            pos_embed = self._position_embedding(mlvl_feats, img2lidars, pad_h, pad_w)
            if head.with_multiview:
                sin_embed = head.positional_encoding(masks)
                sin_embed = head.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(N):
                    xy_embed = head.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = head.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if head.with_multiview:
                pos_embed = head.positional_encoding(masks)
                pos_embed = head.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(N):
                    pe = head.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pe.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = head.reference_points.weight
        query_embeds = head.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(B, 1, 1)

        outs_dec, _ = head.transformer(x, masks, query_embeds, pos_embed, head.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            outputs_class = head.cls_branches[lvl](outs_dec[lvl])
            tmp = head.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        pc = head.pc_range
        all_bbox_preds[..., 0:1] = all_bbox_preds[..., 0:1] * (pc[3] - pc[0]) + pc[0]
        all_bbox_preds[..., 1:2] = all_bbox_preds[..., 1:2] * (pc[4] - pc[1]) + pc[1]
        all_bbox_preds[..., 4:5] = all_bbox_preds[..., 4:5] * (pc[5] - pc[2]) + pc[2]

        return all_cls_scores, all_bbox_preds

    def forward(self, img, img2lidars):
        # img:        (B, N, 3, H, W)
        # img2lidars: (B, N, 4, 4)
        H, W = img.shape[-2], img.shape[-1]
        feats = self._extract_img_feats(img)
        return self._head_forward(feats, img2lidars, pad_h=H, pad_w=W)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type=str,
                        default="projects/configs/petr/petr_r50dcn_gridmask_p4_mini.py")
    parser.add_argument("--checkpoint",
                        type=str,
                        default="ckpts/epoch_24.pth")
    parser.add_argument("--img_height",
                        type=int,
                        default=512)
    parser.add_argument("--img_width",
                        type=int,
                        default=1408)
    parser.add_argument("--num_samples",
                        type=int,
                        default=6)
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--onnx_path",
                        type=str,
                        default="ckpts/epoch_24.onnx")
    parser.add_argument("--input_names",
                        default=["img", "img2lidars"])
    parser.add_argument("--output_names",
                        default=["all_cls_scores", "all_bbox_preds"])
    parser.add_argument("--opset",
                        type=int,
                        default=11)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if hasattr(cfg, "plugin") and cfg.plugin:
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
        else:
            _module_dir = os.path.dirname(args.config)
        _module_path = ".".join(_module_dir.split("/"))
        importlib.import_module(_module_path)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.cuda().eval()

    wrapper = PETRExportWrapper(model).cuda().eval()

    img = torch.randn(
        args.batch_size, args.num_samples, 3, args.img_height, args.img_width
    ).cuda()
    img2lidars = torch.eye(4).expand(args.batch_size, args.num_samples, 4, 4).cuda().contiguous()

    with torch.no_grad():
        torch.onnx.export(wrapper,
                          (img, img2lidars),
                          args.onnx_path,
                          input_names=args.input_names,
                          output_names=args.output_names,
                          opset_version=args.opset,
                          do_constant_folding=False,
                          verbose=False)

    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)

    print("Done")


if __name__ == "__main__":
    main()
