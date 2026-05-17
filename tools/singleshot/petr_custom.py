"""
Create: 2026.05.09
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import io
import torch

import sys

sys.path.append(".")

from tools.singleshot.util import PetrImgMgr
from tools.singleshot.detected_obj import ObjInfo
from tools.singleshot.plt_drawer import plt_drawer
from tools.object_tracker.AD3DMOT import AB3DMOT

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

class petr_custom:
    def __init__(self,
                 model,
                 petrv2=False):
        self.imgMgr = PetrImgMgr()
        self.model = model
        self.img_metas = self.imgMgr.get_img_metas(petrv2)
        self.registered_objs = []
        self.img_size = [1600, 900]
        self.plt_drawer = plt_drawer()
        self.petrv2 = petrv2
        self.cam_objlist = None
        self.bev_objlist = None
        self.prev_obj_info = None
        self.tracking_obj = {}
        self.tracking_id = 0

        self.camera_intrinsic = np.array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
                                          [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
                                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
        self.cls_range_map = {"car": 50,
                              "truck": 50,
                              "bus": 50,
                              "trailer": 50,
                              "construction_vehicle": 50,
                              "pedestrian": 40,
                              "motorcycle": 40,
                              "bicycle": 40,
                              "traffic_cone": 30,
                              "barrier": 30}
        
        self.mot_trackers = {tracking_name: AB3DMOT(0, tracking_name=tracking_name, use_angular_velocity=False, tracking_nuscenes=True) for tracking_name in self.cls_range_map.keys()}

        self.run_time = 0.0

    def doPetr(self,
               cfg,
               img_data,
               threshold=0.25,
               is_trt=True):
        img = self.imgMgr.get_img_data(img_data, cfg.img_norm_cfg, self.petrv2)
        input_data = {}
        input_data["img"] = img[0]
        lidar2img = self.img_metas["lidar2img"]

        img2lidars = []
        img2lidar = []

        for i in range(len(lidar2img)):
            img2lidar.append(np.linalg.inv(lidar2img[i]))

        img2lidars.append(np.asarray(img2lidar))

        img2lidars = torch.from_numpy(np.asarray(img2lidars)).float()

        input_data["img2lidar"] = img2lidars

        match_distance = "iou"
        match_threshold = 0.1
        match_algorithm = "greedy"

        start_ = time.time()

        if not is_trt:
            input_list = []
            input_list.append(input_data["img"].cuda())
            input_list.append(input_data["img2lidar"].cuda())

            with torch.no_grad():
                result = self.model(input_list)
        else:
            result = self.model.inference(img[0], img2lidars)

        detected_obj_list = self.translate_obj_info(result, cfg.class_names, threshold, is_trt)
        tracked_obj_list = self.object_tracking(detected_obj_list, match_distance, match_threshold, match_algorithm)

        self.cam_objlist = self.get_view_visible_obj_list(tracked_obj_list, False)
        self.bev_objlist = self.get_view_visible_obj_list(tracked_obj_list, True)

        self.run_time = time.time() - start_

        print(f"execution time: {self.run_time:.2f} sec")

    def object_tracking(self,
                        detected_obj_list,
                        match_distance,
                        match_threshold,
                        match_algorithm):
        tracked_obj_list = []
        dets = {tracking_name: [] for tracking_name in self.cls_range_map.keys()}
        info = {tracking_name: [] for tracking_name in self.cls_range_map.keys()}

        for objinfo in detected_obj_list:
            angle = objinfo.rotate.angle if objinfo.rotate.axis[2] > 0 else -objinfo.rotate.angle

            detection = np.array([objinfo.size[2], objinfo.size[0], objinfo.size[1], objinfo.center[0], objinfo.center[1], objinfo.center[2], angle])

            information = np.array([objinfo.score])

            dets[objinfo.name].append(detection)
            info[objinfo.name].append(information)

        dets_all = {tracking_name: {"dets": np.array(dets[tracking_name]), "info": np.array(info[tracking_name])} for tracking_name in self.cls_range_map.keys()}

        for label, tracking_name in enumerate(self.cls_range_map.keys()):
            if dets_all[tracking_name]["dets"].shape[0] > 0:
                trackers = self.mot_trackers[tracking_name].update(dets_all[tracking_name], match_distance, match_threshold, match_algorithm)

                for i in range(trackers.shape[0]):
                    objinfo = ObjInfo(list(self.cls_range_map.keys()), trackers[i][8], label, trackers[i][3:6], [trackers[i][1], trackers[i][2], trackers[i][0]], trackers[i][6], trackers[i][7])
                    tracked_obj_list.append(objinfo)

        return tracked_obj_list
    
    def just_draw_same_bbox(self,
                            img_data):
        self.plt_drawer.draw_plot(img_data, self.bev_objlist, self.cam_objlist)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=200)
        buf.seek(0)

        return buf
    
    def get_view_visible_obj_list(self,
                                  objinfo_list,
                                  bev=False):
        tmp_objinfo_list = copy.deepcopy(objinfo_list)

        for objinfo in tmp_objinfo_list:
            if bev:
                objinfo.box2ego_translation()
                corners = objinfo.get_corner()
                points = objinfo.get_view_points(corners, np.eye(4), False)
            else:
                objinfo.ego_pose_record_translation()
                objinfo.calibrated_sensor_translation()

                corners = objinfo.get_corner()
                points = objinfo.get_view_points(corners, self.camera_intrinsic, True)

            objinfo.corners = points

        return tmp_objinfo_list
    
    def translate_obj_info(self,
                           result,
                           class_names,
                           threshold=0.25,
                           is_trt=True):
        if not is_trt:
            scores = result[0][1].cpu().tolist()
            bboxes = LiDARInstance3DBoxes(result[0][0].cpu(), result[0][0].size(-1))
            labels = result[0][2].cpu().int().tolist()
        else:
            scores = result[0].cpu().tolist()
            result[1] = result[1].cpu()
            bboxes = LiDARInstance3DBoxes(result[1], result[1].size(-1))
            labels = result[2].cpu().int().tolist()

        boxes_num = len(scores)

        box_centers = bboxes.gravity_center.detach().numpy()
        box_sizes = bboxes.dims.detach().numpy()
        box_yaws = -bboxes.yaw.detach().numpy() - np.pi / 2

        objinfo_list = []

        for i in range(boxes_num):
            if scores[i] < threshold:
                continue

            objinfo = ObjInfo(class_names, scores[i], labels[i], box_centers[i], box_sizes[i], box_yaws[i])

            objinfo.lidar2ego_translation()

            radius = np.linalg.norm(objinfo.center[:2], 2)

            if radius > self.cls_range_map[objinfo.name]:
                continue

            objinfo.ego2global_translation()
            objinfo_list.append(objinfo)

        return objinfo_list