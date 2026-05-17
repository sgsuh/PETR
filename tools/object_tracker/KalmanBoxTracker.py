"""
Create: 2026.05.05
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import numpy as np

from filterpy.kalman import KalmanFilter
from tools.object_tracker.covariance import Covariance

class KalmanBoxTracker(object):
    count = 0

    def __init__(self,
                 bbox3D,
                 info,
                 covariance_id=0,
                 track_score=None,
                 tracking_name="car",
                 use_angular_velocity=False):
        if not use_angular_velocity:
            self.kf = KalmanFilter(dim_x=10, dim_z=7)
            self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],
                                  [0,1,0,0,0,0,0,0,1,0],
                                  [0,0,1,0,0,0,0,0,0,1],
                                  [0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,1]])
            
            self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0]])
        else:
            self.kf = KalmanFilter(dim_x=11, dim_z=7)
            self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],
                                  [0,1,0,0,0,0,0,0,1,0,0],
                                  [0,0,1,0,0,0,0,0,0,1,0],
                                  [0,0,0,1,0,0,0,0,0,0,1],
                                  [0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,0,1]])
            
            self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0]])
            
        if covariance_id == 0:
            self.kf.P[7:, 7:] *= 1000.
            self.kf.P *= 10.

            self.kf.Q[7:, 7:] *= 0.01
        elif covariance_id == 1:
            covariance = Covariance(covariance_id)
            self.kf.P = covariance.P
            self.kf.Q = covariance.Q
            self.kf.R = covariance.R
        elif covariance_id == 2:
            covariance = Covariance(covariance_id)
            self.kf.P = covariance.P[tracking_name]
            self.kf.Q = covariance.Q[tracking_name]
            self.kf.R = covariance.R[tracking_name]

            if not use_angular_velocity:
                self.kf.P = self.kf.P[:-1, :-1]
                self.kf.Q = self.kf.Q[:-1, :-1]
        else:
            assert(False)

        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info
        self.track_score = track_score
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity

    def update(self,
               bbox3D,
               info):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        if self.still_first:
            self.first_continuing_hit += 1

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2

        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]

        if new_theta >= np.pi:
            new_theta -= np.pi * 2

        if new_theta < -np.pi:
            new_theta += np.pi * 2

        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]

        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
            self.kf.x[3] += np.pi

            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2

            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2

        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        self.info = info

    def predict(self):
        self.kf.predict()

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2

        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
            self.still_first = False

        self.time_since_update += 1
        self.history.append(self.kf.x)

        return self.history[-1]
    
    def get_state(self):
        return self.kf.x[:7].reshape((7, ))            