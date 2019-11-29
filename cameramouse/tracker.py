"""
Author: Toby Baker
Title: Optical Flow Tracker Objects
Date Created: 25/NOV/2019
"""

from camera import WebcamCamera, RealSenseCamera
import cv2
import numpy as np
from time import clock

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
        self.prev_ftrs = cv2.goodFeaturesToTrack(self.prev_gray_img, mask = None, **self.feature_params)
        self.mask = np.zeros_like(self.color_img)

    def update(self):
        """Get new frames"""
        self.gray_img, self.color_img = self.cam.capture_frames()

        """ Get the new location of the points """
        self.update_points()

        ''' Draw the points on the image'''
        self.draw()

        '''Calculate the velocity of the points'''
        self.get_velocity()

        '''Resample the points if necessary'''
        self.resample()

        '''Update previous frames'''
        self.prev_gray_img = self.gray_img

    '''
    Get the new location of the optical flow points
    '''
    def update_points(self):
        # Find the position of the points in the next frame
        self.next, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray_img, self.gray_img, self.prev_ftrs, None, **self.tracker_params)

        if self.next is not None:
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
            if (x_n - x_p < 5) and (x_n - x_p != 0):
                vel_x += x_n - x_p
                x_used += 1
            if (y_n - y_p < 5) and (x_n - x_p != 0):
                vel_y += y_n - y_p
                y_used += 1
        if (x_used != 0 and self.next is not None): self.vel_x = vel_x / x_used
        if (y_used != 0 and self.next is not None): self.vel_y = vel_y / y_used

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

        # moving_points = 0
        # threshold = 1
        # for i, (prev, new) in enumerate(zip(self.prev_points, self.new_points)):
        #     x_p, y_p = prev.ravel()
        #     x_n, y_n = new.ravel()
        #     if (abs(x_p-x_n) > threshold) or (abs(y_p-y_n) > threshold):
        #         moving_points += 1
        #
        # if (len(self.prev_points) < 5):
        #     self.mask = np.zeros_like(self.color_img)

    #
    # def run(self):
    #     try:
    #         while True:
    #             # Get new image
    #             self.gray_img, self.color_img = self.cam.capture_frames()
    #             # For visualisation
    #             vis = self.color_img.copy()
    #             # Get next points
    #             next, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray_img, self.gray_img, self.prev_ftrs, None, **self.tracker_params)
    #             # Selects good feature points for previous position and makes a 2D array
    #             good_old = self.prev_ftrs[status == 1]
    #             # Selects good feature points for next position
    #             good_new = next[status == 1]
    #             # Draws the optical flow tracks
    #             for i, (new, old) in enumerate(zip(good_new, good_old)):
    #                 # Returns a contiguous flattened array as (x, y) coordinates for new point
    #                 a, b = new.ravel()
    #                 # Returns a contiguous flattened array as (x, y) coordinates for old point
    #                 c, d = old.ravel()
    #                 # Draws line between new and old position with green color and 2 thickness
    #                 self.mask = cv2.line(self.mask, (a, b), (c, d), self.color, 2)
    #                 # Draws filled circle (thickness of -1) at new position with green color and radius of 3
    #                 vis = cv2.circle(vis, (a, b), 3, self.color, -1)
    #             # Overlays the optical flow tracks on the original frame
    #             output = cv2.add(vis, self.mask)
    #             # Create  copy of the gray image
    #             self.prev_gray_img = self.gray_img.copy()
    #
    #             # Update the frame ID
    #             # self.frame_idx += 1
    #
    #             #If we have fewer than 5 points moving then resample
    #             moving_points = 0 # the number of moving pixels
    #             threshold = 1 # threshold to be considered moving in pixels
    #             vel_x = 0
    #             vel_y = 0
    #             for i, (prev, new) in enumerate(zip(self.prev_ftrs, good_new.reshape(-1, 1, 2))):
    #                 x_p, y_p = prev.ravel()
    #                 x_n, y_n = new.ravel()
    #                 if (abs(x_p-x_n) > threshold) or (abs(y_p-y_n) > threshold):
    #                     moving_points += 1
    #
    #                 vel_x += x_n - x_p
    #                 vel_y += y_n - y_p
    #
    #             vel_x = vel_x / len(self.prev_ftrs)
    #             vel_y = vel_y / len(self.prev_ftrs)
    #             # print('Previous Points %d' % len(self.prev_ftrs))
    #             # print('Future Points % d' % len(good_new.reshape(-1,1,2)))
    #             #print("X: %.3f, Y: %.3f" % (vel_x, vel_y))
    #
    #
    #             # cv2.putText(self.mask, 'moving points: %d' % moving_points, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.white)
    #             # cv2.putText(self.mask, 'total points: %d' % len(self.prev_ftrs), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.white)
    #
    #
    #             # Updates previous good feature points
    #             self.prev_ftrs = good_new.reshape(-1, 1, 2)
    #
    #             #Every 5 frames update the points that you want to track
    #             # if moving_points < 5:
    #             #     self.mask = np.zeros_like(color_image)
    #             #     self.prev_ftrs = cv2.goodFeaturesToTrack(self.gray_img, mask = None, **self.feature_params)
    #
    #
    #             # Show images
    #             cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #             cv2.imshow('RealSense', output)
    #
    #             # print(self.prev_ftrs.shape)
    #
    #             keyboard = cv2.waitKey(30)
    #             if keyboard == 'q' or keyboard == 27:
    #                 break
    #
    #     finally:
    #         if self.realsense:
    #             pipeline.stop()
    #         cv2.destroyAllWindows()
