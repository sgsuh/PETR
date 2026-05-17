"""
Create: 2026.05.09
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import math
import numpy as np
import pyquaternion

class ObjInfo:
    def __init__(self,
                 class_names,
                 score,
                 label,
                 center,
                 size,
                 yaw,
                 id_=-1):
        self.id_ = int(id_)
        self.name = class_names[label]
        self.score = score
        self.center = center
        self.center_orig = center
        self.size = size
        self.yaw_orig = yaw
        self.rotate = pyquaternion.Quaternion(axis=[0, 0, 1], radians=self.yaw_orig)
        self.velocity = 0
        self.corners = None

        self.color_map = {"car": (70, 130, 180), 
                          "truck": (0, 0, 230), 
                          "bus": (135, 206, 235), 
                          "trailer": (100, 149, 237), 
                          "construction_vehicle": (219, 112, 147), 
                          "pedestrian": (255, 61, 99),
                          "motorcycle": (240, 128, 128),
                          "bicycle": (138, 43, 226),
                          "traffic_cone": (112, 128, 144),
                          "barrier": (210, 105, 30)}
        
        self.color = np.array(self.color_map[self.name]) / 255.0

        self.distance = self.getDistance()

        self.pose_record_rotation = [0.41747928378147564, -0.008941011429559326, 0.003256181928641861, 0.9086366178000811]
        self.pose_record_translation = [732.2886159828824, 948.8005086665738, 0.0]

    def getDistance(self):
        return math.sqrt(self.center_orig[0] ** 2 + self.center_orig[1] ** 2 + self.center_orig[2] ** 2)
    
    def is_same_obj(self,
                    objinfo,
                    tolerence=1):
        diff_distance = math.sqrt((self.center_orig[0] - objinfo.center_orig[0]) ** 2 + (self.center_orig[1] - objinfo.center_orig[1]) ** 2 + (self.center_orig[2] - objinfo.center_orig[2]) ** 2)

        if tolerence < diff_distance or self.name != objinfo.name:
            return False
        
        return True
    
    def rotation(self,
                 quaternion):
        rotation = quaternion
        self.center = np.dot(rotation.rotation_matrix, self.center)
        self.rotate = rotation * self.rotate
        self.velocity = np.dot(rotation.rotation_matrix, self.velocity)

    def translation(self,
                    translation_array):
        self.center += translation_array

    def lidar2ego_translation(self):
        self.rotation(pyquaternion.Quaternion([0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]))
        self.translation(np.array([0.943713, 0.0, 1.84023]))

    def ego2global_translation(self):
        self.rotation(pyquaternion.Quaternion([0.41589926602121, -0.00899084029923763, 0.003201053083100996, 0.9093606097543983]))
        self.translation(np.array([732.0614444472142, 949.0676735861334, 0.0]))

    def ego_pose_record_translation(self):
        self.translation(-np.array(self.pose_record_translation))
        self.rotation(pyquaternion.Quaternion(self.pose_record_rotation).inverse)

    def calibrated_sensor_translation(self):
        self.translation(-np.array([1.70079118954, 0.0159456324149, 1.51095763913]))
        self.rotation(pyquaternion.Quaternion([0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755]).inverse)

    def box2ego_translation(self):
        yaw = pyquaternion.Quaternion(self.pose_record_rotation).yaw_pitch_roll[0]
        self.translation(-np.array(self.pose_record_translation))
        self.rotation(pyquaternion.Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)

    def get_corner(self):
        w, l, h = self.size

        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = np.dot(self.rotate.rotation_matrix, corners)

        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners
    
    def get_view_points(self,
                        corners,
                        view,
                        normalize=True):
        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = corners.shape[1]

        points = np.concatenate((corners, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].reshape(3, 0).reshape(3, nbr_points)

        points = points[:2, :]

        return points