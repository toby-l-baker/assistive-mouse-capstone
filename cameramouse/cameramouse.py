"""
Author: Toby Baker
Title: Camera Mouse
Date Created: 28/NOV/2019
"""

from interface import WindowsMouse, WindowsMonitor, Mouse, Monitor
from camera import RealSenseCamera, WebcamCamera, CameraObject
from tracker import LukasKanadeResampling
from gesture_recognition import KeyboardGestureRecognition, GestureRecognition, Gestures
import numpy as np
import cv2

class CameraMouse():
    def __init__(self):
        self.camera = CameraObject(0)
        self.monitor = Monitor()
        self.mouse = Mouse()
        self.gesture_recognition = GestureRecognition()

    def run(self):
        pass

    def velocity_map(self):
        # TF: https://www.wolframalpha.com/input/?i=plot+tanh%284*x-2%29+%2B+1
        gain_x = 1 # np.tanh(4*self.tracker.vel_x-2) + 1 # hyperbolic function gain can be between 0 and 2
        gain_y = 1 # np.tanh(4*self.tracker.vel_y-2) + 1
        return (gain_x * self.tracker.vel_x, gain_y * self.tracker.vel_y)

    def execute_control(self):
        if self.gesture_recognition.gesture == Gestures.out_of_range:
            return
        else:
            v_x, v_y = self.velocity_map()
            dx = v_x * self.monitor.width #scales velocity to the size of the monitor and divides by 10
            dy = v_y * self.monitor.height
            # print("Current amount to move in pixels: ({}, {})".format(dx, dy))
            self.mouse.move(dx, dy)
            if self.gesture_recognition.gesture == Gestures.click:
                self.mouse.left_click()
            elif self.gesture_recognition.gesture == Gestures.double_click:
                self.mouse.double_click()
            elif self.gesture_recognition.gesture == Gestures.right_click:
                self.mouse.right_click()
            elif self.gesture_recognition.gesture == Gestures.drag:
                self.mouse.drag()

class OpticalFlowMouse(CameraMouse):
    def __init__(self):
        self.camera = RealSenseCamera()
        self.monitor = WindowsMonitor()
        self.mouse = WindowsMouse()
        self.gesture_recognition = KeyboardGestureRecognition()
        self.prev_gray_img, self.color_img = self.camera.capture_frames()
        tracker_params = dict( winSize  = (15, 15), maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict(maxCorners = 500, qualityLevel = 0.2,
                              minDistance = 2, blockSize = 7)
        self.tracker = LukasKanadeResampling(tracker_params, feature_params, self.camera)


    def run(self):
        while True:
            gray_img, color_img = self.camera.capture_frames()
            self.tracker.update(gray_img, color_img)
            # self.gesture_recognition.update()
            self.execute_control()

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
