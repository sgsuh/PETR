"""
Create: 2026.05.09
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import math
import copy
import numpy as np
import pandas as pd
import torch
import mmcv
import cv2
import PIL.Image as pilimg

from mmdet3d.core import LiDARInstance3DBoxes

def is_visible(corners_img,
               corners_3d,
               img_size):
    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < img_size[0])
    visible = np.logical_and(visible, corners_img[1, :] < img_size[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1

    return any(visible) and all(in_front)

class PetrImgMgr:
    def __init__(self):
        self.old_img = None

    def get_img_metas(self,
                      petrv2=False):
        img_metas = {}
        img_metas["pad_shape"] = []

        for i in range(6):
            img_metas["pad_shape"].append((320, 800, 3))

        lidar2img = []
        lidar2img.append(np.array([[6.33137794e+02,  4.08074490e+02,  1.17405010e+01, -1.59077226e+02],
                                   [4.11923896e+00,  1.27531292e+02, -6.30928534e+02, -2.58489066e+02],
                                   [-1.40386752e-04,  9.99826412e-01,  1.86313382e-02, -4.08345062e-01],
                                   [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))
        
        lidar2img.append(np.array([[6.83813281e+02, -3.04664970e+02, -1.46933254e+01, -2.44139475e+02],
                                   [9.16279506e+01,  7.99982646e+01, -6.29669945e+02, -2.85810207e+02],
                                   [8.35612690e-01,  5.49300529e-01,  4.51244948e-03, -5.99209745e-01],
                                   [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))
        
        lidar2img.append(np.array([[2.88655813e+01,  7.57981042e+02,  1.81987664e+01, -1.09125198e+02],
                                   [-8.82185399e+01,  7.85138800e+01, -6.34824496e+02, -2.70956098e+02],
                                   [-8.17283232e-01,  5.76116432e-01,  1.17463061e-02, -4.94588509e-01],
                                   [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))
        
        lidar2img.append(np.array([[-4.07052155e+02, -4.12152780e+02, -7.04047794e+00, -4.26889596e+02],
                                   [3.26232082e+00, -1.07846769e+02, -4.05418921e+02, -2.27592653e+02],
                                   [-5.95219763e-03, -9.99953673e-01, -7.56466193e-03, -1.02865681e+00],
                                   [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))
        
        lidar2img.append(np.array([[-5.74635618e+02,  4.70624955e+02,  4.05316467e+00, -3.11691901e+02],
                                   [-9.78854850e+01, -1.61349889e+01, -6.31311348e+02, -2.04412289e+02],
                                   [-9.48288437e-01, -3.16059480e-01, -2.92479827e-02, -4.41690327e-01],
                                   [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))
        
        lidar2img.append(np.array([[1.50542084e+02, -7.32070374e+02, -3.02997996e+01, -1.78536230e+02],
                                   [1.09068889e+02, -1.79212929e+01, -6.31601436e+02, -2.33368243e+02],
                                   [9.33277897e-01, -3.58619863e-01, -1.95999939e-02, -5.04299162e-01],
                                   [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))
        
        if petrv2:
            for i in range(6):
                lidar2img.append(lidar2img[i])
                img_metas["pad_shape"].append(img_metas["pad_shape"][i])

        img_metas["img_shape"] = img_metas["pad_shape"]
        img_metas["lidar2img"] = lidar2img
        img_metas["box_type_3d"] = LiDARInstance3DBoxes
        img_metas["timestamp"] = [0.03567790985107422, 0.028186798095703125, 0.043226003646850586, 0.01057291030883789, 0.0008478164672851562, 0.020329952239990234, 1.2856779098510742, 1.2781867980957031, 1.2932260036468506, 1.260572910308838, 1.2508478164672852, 1.2703299522399902]

        return img_metas
    
    def get_img_data(self,
                     img_path,
                     img_norm_cfg,
                     petrv2=False):
        img_tmp = img_path

        imgobj = pilimg.fromarray(img_tmp)

        imgobj = imgobj.resize((800, 450))
        imgobj = imgobj.crop((0, 130, 800, 450))
        imgobj = imgobj.rotate(0)
        img_tmp = np.array(imgobj)
        img_tmp = img_tmp.astype(np.float32)

        stdinv = 1 / np.float64(img_norm_cfg["std"]).reshape(1, -1)

        cv2.subtract(img_tmp, np.float64(img_norm_cfg["mean"]).reshape(1, -1), img_tmp)
        cv2.multiply(img_tmp, stdinv, img_tmp)

        divisor = 32
        pad_h = int(np.ceil(img_tmp.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img_tmp.shape[1] / divisor)) * divisor
        padding = (0, 0, pad_w - img_tmp.shape[1], pad_h - img_tmp.shape[0])

        img_tmp = cv2.copyMakeBorder(img_tmp, padding[1], padding[3], padding[0], padding[2], 0, value=0)
        img_tmp = img_tmp.transpose(2, 0, 1)
        img = np.expand_dims(img_tmp, axis=0)
        img_input = copy.deepcopy(img)
        zeros = np.zeros(img_input.shape)

        for i in range(5):
            img_input = np.append(img_input, zeros, axis=0)

        if petrv2:
            if self.old_img is None:
                img_input = np.append(img_input, img_input, axis=0)
            else:
                img_input = np.append(img_input, self.old_img, axis=0)

                for i in range(5):
                    img_input = np.append(img_input, zeros, axis=0)

        img_input = np.expand_dims(img_input, axis=0)
        img_input = np.expand_dims(img_input, axis=0)
        img_input = torch.Tensor(img_input)

        self.old_img = copy.deepcopy(img)

        return img_input
