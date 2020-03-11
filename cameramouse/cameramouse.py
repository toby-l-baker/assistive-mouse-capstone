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
import cv2, sys, argparse
sys.path.append('../hand_tracking')
from hand_tracking import HandTracker

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
            dx = v_x # * self.monitor.width #scales velocity to the size of the monitor and divides by 10
            dy = v_y # * self.monitor.height
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
    def __init__(self, camera):
        self.camera = camera
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

class HandSegmentationMouse(CameraMouse):
    def __init__(self, camera, filter, filter_size):
        self.camera = camera
        self.monitor = WindowsMonitor()
        self.mouse = WindowsMouse()
        self.gesture_recognition = KeyboardGestureRecognition()
        self.tracker = HandTracker(camera, filter_size, filter)
        self.lin_term = 1/30
        self.quad_term = 1/30000
        self.lin_sens = 5
        self.quad_sens = 5

    def updateSens(self, _):
        self.lin_sens = cv2.getTrackbarPos("lin term", "FeedMe")
        self.quad_sens = cv2.getTrackbarPos("quad term", "FeedMe")

    def run(self):
        # i = 0
        flag = False
        cv2.namedWindow("FeedMe")
        cv2.createTrackbar("quad term", "FeedMe", \
          self.lin_sens, 10, self.updateSens)
        cv2.createTrackbar("lin term", "FeedMe", \
          self.quad_sens, 10, self.updateSens)
        while True:
            # grab frames
            gray_frame, color_frame = self.camera.capture_frames()

            if not self.tracker.found: # hand is lost
                if flag: # only execute once
                    self.tracker.handSeg.get_histogram()
                    flag = False
                self.tracker.global_recognition(color_frame)
            else: # found the hand lets track it
                self.tracker.get_velocity(color_frame)
                self.execute_control()
                flag = True

            cv2.imshow("FeedMe", color_frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                cv2.destroyWindow("FeedMe")
                break

    def velocity_map(self):
        # TF: https://www.wolframalpha.com/input/?i=plot+tanh%284*x-2%29+%2B+1
        # print(self.tracker.vel_x)
        def get_gain(vel):
            return self.lin_term*self.lin_sens + self.quad_term*self.quad_sens*abs(vel)

        ret_x = int(self.tracker.vel_x * get_gain(self.tracker.vel_x))
        ret_y = int(self.tracker.vel_y * get_gain(self.tracker.vel_y))
        return ret_x, ret_y
