"""
Create: 2022.10.24
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import torch
import torch.nn.functional as F
import math
from mmcv.runner import force_fp32
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.models.dense_heads import PETRHead

def pos2posemb3d(pos,
                 num_pos_feats=128,
                 temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos.unsqueeze(2) / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_x[:, 1, :], pos_x[:, 0, :], pos_x[:, 2, :]), dim=-1)

    return posemb

@HEADS.register_module()
class PETRHeadExport(PETRHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(type="CrossEntropyLoss", bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0),
                 loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                 loss_iou=dict(type="GIoULoss", loss_weight=2.0),
                 train_cfg=dict(assigner=dict(type="HungarianAssigner",
                                              cls_cost=dict(type="ClassificationCost", weight=1.),
                                              reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                                              iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1,
                 position_range=[-65, -65, -8,0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         num_query,
                         num_reg_fcs,
                         transformer,
                         sync_cls_avg_factor,
                         positional_encoding,
                         code_weights,
                         bbox_coder,
                         loss_cls,
                         loss_bbox,
                         loss_iou,
                         train_cfg,
                         test_cfg,
                         with_position,
                         with_multiview,
                         depth_step,
                         depth_num,
                         LID,
                         depth_start,
                         position_range,
                         init_cfg,
                         normedlinear,
                         **kwargs)
        
        if "img_height" in kwargs:
            self.img_height = kwargs["img_height"]
        else:
            self.img_height = 320

        if "img_width" in kwargs:
            self.img_width = kwargs["img_width"]
        else:
            self.img_width = 800

    def position_embeding(self,
                          img_feats,
                          lidar2img,
                          masks=None):
        eps = 1e-5
        pad_h, pad_w = self.img_height, self.img_width
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.clamp(coords[..., 2:3], min=eps)
        img2lidars = lidar2img
        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])
        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask
    
    def forward(self,
                mlvl_feats,
                lidar2img):
        x = mlvl_feats[0]
        batch_size, num_cams = x.size(0), x.size(1)
        input_img_h, input_img_w = self.img_height, self.img_width
        masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w = self.img_height, self.img_width
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, lidar2img, masks)
            pos_embed = coords_position_embeding

            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []

                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeeze(1))

                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []

                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))

                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())

            assert reference.shape[-1] == 3

            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        outs = {"all_cls_scores": all_cls_scores,
                "all_bbox_preds": all_bbox_preds,
                "enc_cls_scores": None,
                "enc_bbox_preds": None}
        
        return outs
    
    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self,
                   preds_dicts):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []

        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            scores = preds["scores"]
            labels = preds["labels"].float()
            ret_list.append([bboxes, scores, labels])

        return ret_list

    def get_bboxes_export(self, preds_dicts):
        """ONNX-friendly batched decode.

        Mirrors NMSFreeCoder.decode_single but keeps static (max_num, 9)
        shapes: no boolean post_center_range mask, no Python-side per-sample
        list. denormalize_bbox is inlined for the 10D normalized PETR format
        ([cx, cy, w, l, cz, h, sin, cos, vx, vy]) to avoid the size>8 branch
        that produces an IIfConditional in the ONNX graph (TRT 8.5 rejects).
        Z is shifted from gravity-center to bottom-center to match
        LiDARInstance3DBoxes(origin=(0.5, 0.5, 0)).
        """
        cls_scores_all = preds_dicts["all_cls_scores"][-1]
        bbox_preds_all = preds_dicts["all_bbox_preds"][-1]
        max_num = self.bbox_coder.max_num
        num_classes = self.bbox_coder.num_classes
        batch_size = cls_scores_all.size(0)

        out_bboxes, out_scores, out_labels = [], [], []
        for i in range(batch_size):
            cls_scores = cls_scores_all[i].sigmoid()
            scores, indexs = cls_scores.view(-1).topk(max_num)
            labels = indexs % num_classes
            bbox_index = indexs // num_classes
            bp = bbox_preds_all[i][bbox_index]

            cx = bp[..., 0:1]
            cy = bp[..., 1:2]
            w = bp[..., 2:3].exp()
            l = bp[..., 3:4].exp()
            cz = bp[..., 4:5]
            h = bp[..., 5:6].exp()
            rot = torch.atan2(bp[..., 6:7], bp[..., 7:8])
            vx = bp[..., 8:9]
            vy = bp[..., 9:10]
            cz_bottom = cz - h * 0.5
            bboxes = torch.cat([cx, cy, cz_bottom, w, l, h, rot, vx, vy], dim=-1)

            out_bboxes.append(bboxes)
            out_scores.append(scores)
            out_labels.append(labels.float())

        return (torch.stack(out_bboxes, dim=0),
                torch.stack(out_scores, dim=0),
                torch.stack(out_labels, dim=0))