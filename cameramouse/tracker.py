"""
Author: Toby Baker
Title: Optical Flow Tracker Objects
Date Created: 25/NOV/2019
"""

from camera import WebcamCamera, RealSenseCamera
import cv2
import numpy as np
import time

class TrackerObject():
    """
    Object for an instantiation of a Various Optical Flow Trackers
    Inputs:
        realsense: True if using a realsense
        src: video source when using webcam/built-in camera
        tracker_params: parameters for the tracker
        feature_params: parameters for the feature detector (Shi-Tomasi in this case)
    """
    def __init__(self, tracker_params, feature_params, camera, src=0):
        # Parameters for Lucas-Kanade optical flow
        self.tracker_params = tracker_params
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = feature_params

        # Variable for color to draw optical flow track
        self.color = (0, 255, 0)
        self.white = (255, 255, 255)

        # Use either the realsense or webcam as the src
        self.cam = camera



    """
    To run the tracker you pass it a set of points to track, it will then output
    the points of those locations in the next frames in addition to a boolean indicating
    whether the points should be resampled or not.
    """
    def update(self):
        pass

    def draw(self):
        pass

    def get_velocity(self):
        pass

class LukasKanadeResampling(TrackerObject):
    """
    Implementation of the Lucas-Kanade Optical Flow Tracker with resampling.
    """
    def __init__(self, tracker_params, feature_params, camera):
        super().__init__(tracker_params, feature_params, camera)
        self.prev_gray_img, self.color_img = self.cam.capture_frames()
        self.old_t = time.time()
        self.width, self.height = self.prev_gray_img.shape
        self.prev_ftrs = cv2.goodFeaturesToTrack(self.prev_gray_img, mask = None, **self.feature_params)
        self.mask = np.zeros_like(self.color_img)

    def update(self, gray_img, color_img):
        """Get new frames"""
        self.gray_img = gray_img
        self.color_img = color_img
        self.new_t = time.time()

        """ Get the new location of the points """
        self.update_points()

        """ Draw the points on the image"""
        self.draw()

        '''Calculate the velocity of the points'''
        self.get_velocity()

        '''Resample the points if necessary'''
        self.resample()

        '''Update previous frames'''
        self.prev_gray_img = self.gray_img
        self.old_t = self.new_t

    '''
    Get the new location of the optical flow points
    '''
    def update_points(self):
        # Find the position of the points in the next frame
        self.next, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray_img, self.gray_img, self.prev_ftrs, None, **self.tracker_params)

        if self.next is not None: # prevent errors when next is None, the points will be resampled later if next is none
            self.prev_points = self.prev_ftrs[status == 1]
            self.new_points = self.next[status == 1]

    '''
    Get the velocity of the optical flow points attached to the hand
    '''
    def get_velocity(self):
        vel_x = 0
        vel_y = 0
        x_used = 0
        y_used = 0
        for i, (prev, new) in enumerate(zip(self.prev_points, self.new_points)):
            x_p, y_p = prev.ravel()
            x_n, y_n = new.ravel()
            if (abs(x_n - x_p) < 10) and (abs(x_n - x_p) >= 1): # filter out huge movements and tiny movements
                vel_x += x_n - x_p
                x_used += 1
            if (abs(y_n - y_p) < 10) and (abs(x_n - x_p) >= 1): # filter out huge movements and tiny movements
                vel_y += y_n - y_p
                y_used += 1
        dt = self.new_t - self.old_t
        if (x_used != 0 and self.next is not None): # calculate velocity of used points
            self.vel_x = vel_x / x_used / self.width / (dt)
        else: # if there are no points set vel to zero
            self.vel_x = 0
        if (y_used != 0 and self.next is not None): # calculate velocity of used points
            self.vel_y = vel_y / y_used / self.height / (dt)
        else: # if there are no points set vel to zero
            self.vel_y = 0

    '''
    Draw the motion of the optical flow points on the coloured image
    '''
    def draw(self):
        vis = self.color_img.copy()
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(self.new_points.reshape(-1,1,2), self.prev_points.reshape(-1,1,2))):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            self.mask = cv2.line(self.mask, (a, b), (c, d), self.color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            vis = cv2.circle(vis, (a, b), 3, self.color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(vis, self.mask)

        # Show images
        cv2.namedWindow('Optical Flow Tracker', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Optical Flow Tracker', output)

    '''
    Resample the features to track if they are no longer good or there are none
    to tracks
    '''
    def resample(self):
        if self.next is None:
            self.mask = np.zeros_like(self.color_img)
            self.prev_ftrs = cv2.goodFeaturesToTrack(self.gray_img, mask = None, **self.feature_params)
            # TODO: change mask to be the region of interest in the image
        else:
            self.prev_ftrs = self.new_points.reshape(-1, 1, 2)
