"""
Create: 2026.05.05
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import numpy as np
import copy

from scipy.spatial import ConvexHull

from tools.object_tracker.linear_assignment import linear_assignment
from tools.object_tracker.KalmanBoxTracker import KalmanBoxTracker

def angle_in_range(angle):
    if angle > np.pi:
        angle -= 2 * np.pi

    if angle < -np.pi:
        angle += 2 * np.pi

    return angle

def diff_orientation_correction(det,
                                trk):
    diff = det - trk
    diff = angle_in_range(diff)

    if diff > np.pi / 2:
        diff -= np.pi

    if diff < -np.pi / 2:
        diff += np.pi

    diff = angle_in_range(diff)

    return diff

def poly_area(x,
              y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def box3d_vol(corners):
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))

    return a * b * c

def polygon_clip(subjectPolygon, 
                 clipPolygon):
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
    
    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])

        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]
    
    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex

            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())

                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())

            s = e

        cp1 = cp2

        if len(outputList) == 0:
            return None
        
    return outputList

def convex_hull_intersection(p1,
                             p2):
    inter_p = polygon_clip(p1, p2)

    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)

        return inter_p, hull_inter.volume
    else:
        return None, 0.0
    
def iou3d(corners1,
          corners2):
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)

    return iou, iou_2d

def greedy_match(distance_matrix):
    matched_indices = []

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks

    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])

        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)

    return matched_indices

def associate_detections_to_trackers(detections,
                                     trackers,
                                     iou_threshold=0.1,
                                     use_mahalanobis=False,
                                     dets=None,
                                     trks=None,
                                     trks_S=None,
                                     mahalanobis_threshold=0.1,
                                     print_debug=False,
                                     match_algorithm="greedy"):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)
    
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    distance_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    if use_mahalanobis:
        assert dets is not None
        assert trks is not None
        assert trks_S is not None

    if use_mahalanobis and print_debug:
        print("dets.shape: ", dets.shape)
        print("dets: ", dets)
        print("trks.shape: ", trks.shape)
        print("trks: ", trks)
        print("trks_S.shape: ", trks_S.shape)
        print("trks_S: ", trks_S)

        S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]
        S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]

        print("S_inv_diag: ", S_inv_diag)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if use_mahalanobis:
                S_inv = np.linalg.inv(trks_S[t])
                diff = np.expand_dims(dets[d] - trks[t], axis=1)

                corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
                diff[3] = corrected_angle_diff
                distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
            else:
                iou_matrix[d, t] = iou3d(det, trk)[0]
                distance_matrix = -iou_matrix

    if match_algorithm == "greedy":
        matched_indices = greedy_match(distance_matrix)
    elif match_algorithm == "pre_threshold":
        if use_mahalanobis:
            to_max_mask = distance_matrix > mahalanobis_threshold
            distance_matrix[to_max_mask] = mahalanobis_threshold + 1
        else:
            to_max_mask = iou_matrix < iou_threshold
            distance_matrix[to_max_mask] = 0
            iou_matrix[to_max_mask] = 0

        matched_indices = linear_assignment(distance_matrix)
    else:
        matched_indices = linear_assignment(distance_matrix)

    if print_debug:
        print("distance_matrix.shape: ", distance_matrix.shape)
        print("distance_matrix: ", distance_matrix)
        print("matched_indices: ", matched_indices)

    unmatched_detections = []

    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []

    for t, trk in enumerate(trackers):
        if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []

    for m in matched_indices:
        match = True

        if use_mahalanobis:
            if distance_matrix[m[0], m[1]] > mahalanobis_threshold:
                match = False
        else:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                match = False

        if not match:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    if print_debug:
        print("matches: ", matches)
        print("unmatched_detections: ", unmatched_detections)
        print("unmatched_trackers: ", unmatched_trackers)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def roty(t):
    c = np.cos(t)
    s = np.sin(t)

    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def convert_3dbox_to_8corner(bbox3d_input,
                              nuscenes_to_kitti=False):
    bbox3d = copy.copy(bbox3d_input)

    if nuscenes_to_kitti:
        bbox3d_nuscenes = copy.copy(bbox3d)

        bbox3d[0] = bbox3d_nuscenes[1]
        bbox3d[1] = -bbox3d_nuscenes[2]
        bbox3d[2] = -bbox3d_nuscenes[0]
        bbox3d[3] = -bbox3d_nuscenes[3]
        bbox3d[4] = bbox3d_nuscenes[5]
        bbox3d[5] = bbox3d_nuscenes[4]

    R = roty(bbox3d[3])

    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)

class AB3DMOT(object):
    def __init__(self,
                 covariance_id=0,
                 max_age=2,
                 min_hits=3,
                 tracking_name="car",
                 use_angular_velocity=False,
                 tracking_nuscenes=False):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.covariance_id = covariance_id
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.tracking_nuscenes = tracking_nuscenes

    def update(self,
               dets_all,
               match_distance,
               match_threshold,
               match_algorithm):
        dets, info = dets_all["dets"], dets_all["info"]

        dets = dets[:, self.reorder]

        self.frame_count += 1

        print_debug = False

        if print_debug:
            for trk_tmp in self.trackers:
                print("trk_tmp.id: ", trk_tmp.id)

        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]

            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        if print_debug:
            for trk_tmp in self.trackers:
                print("trk_tmp.id: ", trk_tmp.id)

        dets_8corner = [convert_3dbox_to_8corner(det_tmp, match_distance == "iou" and self.tracking_nuscenes) for det_tmp in dets]

        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []

        trks_8corner = [convert_3dbox_to_8corner(trk_tmp, match_distance == "iou" and self.tracking_nuscenes) for trk_tmp in trks]
        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]

        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)
            trks_S = np.stack(trks_S, axis=0)

        if match_distance == "iou":
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner,
                                                                                       trks_8corner,
                                                                                       iou_threshold=match_threshold,
                                                                                       print_debug=print_debug,
                                                                                       match_algorithm=match_algorithm)
        else:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner,
                                                                                       trks_8corner,
                                                                                       use_mahalanobis=True,
                                                                                       dets=dets,
                                                                                       trks=trks,
                                                                                       trks_S=trks_S,
                                                                                       mahalanobis_threshold=match_threshold,
                                                                                       print_debug=print_debug,
                                                                                       match_algorithm=match_algorithm)
        
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0], info[d, :][0])
                detection_score = info[d, :][0][-1]
                trk.track_score = detection_score

        for i in unmatched_dets:
            detection_score = info[i][-1]
            track_score = detection_score
            trk = KalmanBoxTracker(dets[i, :],
                                   info[i, :],
                                   self.covariance_id,
                                   track_score,
                                   self.tracking_name,
                                   False)
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()
            d = d[self.reorder_back]

            if (trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1, -1))

            i -= 1

            if trk.time_since_update >= self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        
        return np.empty((0, 15 + 7))