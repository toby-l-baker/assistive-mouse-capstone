"""
Author: Toby Baker
Title: Optical Flow Tracker
Date Created: 25/NOV/2019
"""

from camera import WebcamCamera, RealSenseCamera
import cv2
import numpy as np
from time import clock

class Tracker():
    """
    Class for an instantiation of a Lucas-Kanade Optical Flow Tracker
    """
    def __init__(self, realsense=True, src=0):
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict( maxCorners = 500,
                               qualityLevel = 0.2,
                               minDistance = 2,
                               blockSize = 7 )

        # Variable for color to draw optical flow track
        self.color = (0, 255, 0)

        # Tracker parameters
        self.track_len = 10 #number of points to track
        self.detect_interval  = 5 #how often to redetect points
        self.tracks = [] #points to track
        self.frame_idx = 0 #current frame number

        # Use either the realsense or webcam as the src
        if realsense:
            self.cam = RealSenseCamera()
        else:
            self.cam = WebcamCamera(src)

        # Capture initial frames and mask
        self.prev_img, self.color_img = self.cam.capture_frames()
        self.prev_ftrs = cv2.goodFeaturesToTrack(self.prev_img, mask = None, **self.feature_params)
        self.mask = np.zeros_like(self.color_img)

    def run_back_projection(self):
        while True:
            new_img, color_img = self.cam.capture_frames()
            vis = color_img.copy()
            # if we have points to track then do so
            if len(self.tracks) > 0:
                prev_img = self.prev_img
                # get most recent points from tracks and store in p0
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # get new set of points basef off optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img, new_img, p0, None, **self.lk_params)
                # get points from back projection
                p0r, st, err = cv2.calcOpticalFlowPyrLK(new_img, prev_img, p1, None, **self.lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                cv2.putText(vis, 'track count: %d' % len(self.tracks), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))

            if self.frame_idx % self.detect_interval == 0:
                self.mask = np.zeros_like(new_img)
                self.mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(self.mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(new_img, mask = self.mask, **self.feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = new_img.copy()
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
    def run_standard(self):
        try:
            while True:
                # Get new image
                gray_image, color_image = self.cam.capture_frames()
                # For visualisation
                vis = color_image.copy()
                # Get next points
                next, status, error = cv2.calcOpticalFlowPyrLK(self.prev_img, gray_image, self.prev_ftrs, None, **self.lk_params)
                # Selects good feature points for previous position
                good_old = self.prev_ftrs[status == 1]
                # Selects good feature points for next position
                good_new = next[status == 1]
                # Draws the optical flow tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    # Returns a contiguous flattened array as (x, y) coordinates for new point
                    a, b = new.ravel()
                    # Returns a contiguous flattened array as (x, y) coordinates for old point
                    c, d = old.ravel()
                    # Draws line between new and old position with green color and 2 thickness
                    self.mask = cv2.line(self.mask, (a, b), (c, d), (0,255,0), 2)
                    # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                    vis = cv2.circle(vis, (a, b), 3, (0,255,0), -1)
                # Overlays the optical flow tracks on the original frame
                output = cv2.add(vis, self.mask)
                # Create  copy of the gray image
                self.prev_img = gray_image.copy()

                # Updates previous good feature points
                self.prev_ftrs = good_new.reshape(-1, 1, 2)

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                # images = np.hstack((color_image, depth_colormap))
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', output)

                keyboard = cv2.waitKey(30)
                if keyboard == 'q' or keyboard == 27:
                    break

        finally:
            # if realsense:
            #     pipeline.stop()
            cv2.destroyAllWindows()
