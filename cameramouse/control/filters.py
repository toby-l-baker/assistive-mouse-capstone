"""
Control Module

Contents:
Filters: FIR and IIR filters to reduce noise
"""

import time, copy
import numpy as np
import gesture_recognition

class Filter():
    def __init__(self, size):
        self.positions = np.ones((size, 2))
        self.filter_size = size


    """Push a new detected centroid onto the stack"""
    def insert_point(self, point):
        self.positions[1:,:] = self.positions[:-1,:]
        self.positions[0, :] = point

    def get_filtered_position(self):
        pass

class FIRFilter(Filter):
    def __init__(self, size):
        super().__init__(size)

    """Returns the averaged position of our x most recent detections"""
    def get_filtered_position(self, centroid):
        self.insert_point(centroid)
        p_x = np.average(self.positions[:, 0])
        p_y = np.average(self.positions[:, 1])
        return np.array([int(p_x), int(p_y)])

class IIRFilter(Filter):
    def __init__(self, size):
        super().__init__(size)
        self.alpha = 0.7 # how much to weight old readings

    def get_filtered_position(self, centroid):
        frac = self.alpha/self.filter_size
        p_x = np.sum(frac*self.positions[:, 0]) + (1-self.alpha)*centroid[0]
        p_y =  np.sum(frac*self.positions[:, 1])  + (1-self.alpha)*centroid[1]
        self.insert_point(np.array([p_x, p_y]))
        return np.array([p_x, p_y])