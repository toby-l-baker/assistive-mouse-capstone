"""
Author: Toby Baker
Title: Camera Objects to allow for testing of multiple cameras
Date Created: 28 Nov 2018
"""

import numpy as np
import cv2
import pyrealsense2 as rs


class CameraObject():
    """
    Camera object to read frames and process them to get points to track
    """
    def __init__(self):
        pass

    def capture_color_frame(self):
        pass

    def capture_gray_frame(self):
        img = self.capture_color_frame()
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def capture_frames(self):
        img = self.capture_color_frame()
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img)

class WebcamCamera(CameraObject):
    def __init__(self, src):
        super().__init__()
        self.cam = cv2.VideoCapture(src)
        frame = self.capture_color_frame()
        self.width, self.height, _ = frame.shape
        self.im_shape = (self.width, self.height)
        print("[DEBUG] Image Resolution: Width {}, Height {}".format(self.width, self.height))
        print("[DEBUG] Initialised Webcam")

    def capture_color_frame(self):
        ret, frame = self.cam.read()
        return frame


class RealSenseCamera(CameraObject):
    def __init__(self, color=True, depth=False):
        super().__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if depth:
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        if color:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        frame = self.capture_color_frame()
        self.height, self.width, _ = frame.shape
        self.im_shape = (self.width, self.height)
        print("[DEBUG] Image Resolution: Width {}, Height {}".format(self.width, self.height))
        print("[DEBUG] Initialised Realsense")

    def capture_color_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
