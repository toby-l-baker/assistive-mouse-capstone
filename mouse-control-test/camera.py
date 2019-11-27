"""
Author: Toby Baker
Title: Camera class
Date Created: 25/NOV/2019
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
        print("Initialised Webcam")

    def capture_color_frame(self):
        ret, frame = self.cam.read()
        return frame


class RealSenseCamera(CameraObject):
    def __init__(self, color=True, depth=False):
        super().__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if depth:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if color:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        print("Initialised Webcam")

    def capture_color_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
