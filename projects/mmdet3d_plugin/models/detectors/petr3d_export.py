"""
Create: 2022.10.24
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import torch
from mmdet.models import DETECTORS
from projects.mmdet3d_plugin.models.detectors import Petr3D

@DETECTORS.register_module()
class Petr3DExport(Petr3D):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Petr3DExport, self).__init__(use_grid_mask,
                                           pts_voxel_layer,
                                           pts_voxel_encoder,
                                           pts_middle_encoder,
                                           pts_fusion_layer,
                                           img_backbone,
                                           pts_backbone,
                                           img_neck,
                                           pts_neck,
                                           pts_bbox_head,
                                           img_roi_head,
                                           img_rpn_head,
                                           train_cfg,
                                           test_cfg,
                                           pretrained)
        
    def extract_img_feat(self,
                         img):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)

        if img is not None:
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)

            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []

        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped
    
    def forward(self,
                data):
        img = data[0]
        img2lidar = data[1]
        img_feats = self.extract_img_feat(img)
        outs = self.pts_bbox_head(img_feats, img2lidar)
        bbox_result = self.pts_bbox_head.get_bboxes(outs)

        return bbox_result