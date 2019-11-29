"""
Author: Toby Baker
Title: Camera Mouse
Date Created: 28/NOV/2019
"""

from interface import WindowsMouse, WindowsMonitor
from camera import RealSenseCamera, WebcamCamera
from tracker import LukasKanadeResampling
import cv2

class CameraMouse():
    def __init__(self):
        pass

class OpticalFlowMouse(CameraMouse):
    def __init__(self):
        self.camera = WebcamCamera(0)
        self.monitor = WindowsMonitor()
        self.mouse = WindowsMouse()
        self.prev_gray_img, self.color_img = self.camera.capture_frames()
        tracker_params = dict( winSize  = (15, 15), maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict(maxCorners = 500, qualityLevel = 0.2,
                              minDistance = 2, blockSize = 7)
        self.tracker = LukasKanadeResampling(tracker_params, feature_params, self.camera)

    def run(self):
        while True:
            self.tracker.update()
            self.mouse.move(self.tracker.vel_x, self.tracker.vel_y)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
