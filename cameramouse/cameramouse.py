"""
Author: Toby Baker
Title: Camera Mouse
Date Created: 28/Nov/2019

Description: Full implementation of a computer mouse including cursor movement, and mouse actions.

This was intended to be as interchangable as possible so different control, filters, segmentation and gesture recognition techniques can be tested

"""

import numpy as np
import cv2, sys, argparse
from gesture_recognition.gestures import Gestures
import utils

try:
    import win32api
except:
    pass

class CameraMouse():
    def __init__(self, opts):
        self.opts = opts
        self.monitor = utils.Loader.load_monitor(opts["os"])
        self.mouse = utils.Loader.load_mouse(opts["os"])
        self.camera = utils.Loader.load_camera(opts["camera"])
        self.gesture_recognition = utils.Loader.load_gesture(opts["gesture"])
        self.tracker = utils.Loader.load_tracker(opts["tracking"], self.camera)
        self.control = utils.Loader.load_control(opts["control"], self.mouse, self.monitor, self.camera.im_shape)
        self.last_gesture_update = 0

    def run(self):
        cv2.namedWindow("CameraMouse")

        while True:
            # grab frames
            gray_frame, color_frame = self.camera.capture_frames()

            # if the hand is lost, recalibrate and locate the hand
            if not self.tracker.found:
                initial_pos = self.tracker.calibrate(frame)
                if initial_pos is not None:
                    self.control.setup(initial_pos)
    
            # hand is still in view, track it
            else:
                centroid = self.tracker.update_position(color_frame, self.opts["control"]["type"])
                cur_gesture = self.gesture_recognition.update()

                # Make sure we aren't doubling down on the same action
                if self.last_gesture_update < self.gesture_recognition.i:
                    self.last_gesture_update = self.gesture_recognition.i
                elif self.last_gesture_update == self.gesture_recognition.i and not (cur_gesture == Gestures.drag):
                    cur_gesture = Gestures.null
                
                self.control.update(centroid)
                self.control.execute(cur_gesture)

            cv2.imshow("CameraMouse", color_frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                cv2.destroyWindow("CameraMouse")
                break

class HandSegmentationMouse(CameraMouse):
    def __init__(self, opts):
        super().__init__(opts)

    def run(self):
        calibrated = False
        cv2.namedWindow("CameraMouse")

        while True:

            # grab frames
            gray_frame, color_frame = self.camera.capture_frames()

            # if the hand is lost, recalibrate and locate the hand
            if not self.tracker.found:
                if calibrated == False:
                    self.tracker.handSeg.get_histogram()
                    calibrated = True
                initial_pos = self.tracker.global_recognition(color_frame)
                if initial_pos is not None:
                    self.control.setup(initial_pos)
    
            # hand is still in view, track it
            else:
                centroid = self.tracker.update_position(color_frame, self.opts["control"]["type"])
                # gesture = self.gesture_recognition.update()

                # Make sure we aren't doubling down on the same action
                if self.last_gesture_update < self.gesture_recognition.i:
                    self.last_gesture_update = self.gesture_recognition.i
                elif self.last_gesture_update == self.gesture_recognition.i and not (self.gesture_recognition.gesture == Gestures.drag):
                    self.gesture_recognition.gesture = Gestures.null
                
                self.control.update(centroid)
                self.control.execute(self.gesture_recognition.gesture)

                caibrated = False # to ensure if the hand is lost we recalibrate

            cv2.imshow("CameraMouse", color_frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                cv2.destroyWindow("CameraMouse")
                break
