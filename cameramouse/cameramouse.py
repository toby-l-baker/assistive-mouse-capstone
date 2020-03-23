"""
Author: Toby Baker
Title: Camera Mouse
Date Created: 28/NOV/2019
"""

# from interface import WindowsMouse, WindowsMonitor, WindowsMouse, WindowsMouse
from camera import RealSenseCamera, WebcamCamera, CameraObject
from tracker import LukasKanadeResampling
from gesture_recognition import KeyboardGestureRecognition, GestureRecognition, Gestures
import numpy as np
import cv2, sys, argparse
sys.path.append('../hand_tracking')
from hand_tracking import HandTracker

try:
    import win32api
except:
    print("Cannot Import win32api")

class CameraMouse():
    def __init__(self, monitor, mouse, camera):
        self.camera = camera
        self.monitor = monitor
        self.mouse = mouse
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
            x, y = 0, 0
            dx, dy = 0, 0
            if self.control == "vel":
                dx, dy = self.velocity_map()
                cx, cy = self.mouse.position()
                x, y = cx+dx, cy+dy
            elif self.control == "abs":
                x, y = self.position_map()
                cx, cy = self.mouse.position()
                dx, dy = x - cx, y - cy
            if self.gesture_recognition.gesture == Gestures.drag:
                if self.mouse.state == "UP":
                    self.mouse.mouse_down()
                    self.mouse.state = "DOWN"
                else:
                    self.mouse.moveD(dx, dy) # needs to use differences
                    pass
            else:
                if self.mouse.state == "DOWN":
                    self.mouse.mouse_up()
                    self.mouse.state = "UP"
                if self.gesture_recognition.gesture == Gestures.click:
                    self.mouse.left_click()
                elif self.gesture_recognition.gesture == Gestures.double_click:
                    self.mouse.double_click()
                elif self.gesture_recognition.gesture == Gestures.right_click:
                    self.mouse.right_click()
                else:
                    # print("MOVING TO {} {}".format(x, y))
                    self.mouse.move(x, y)

# class OpticalFlowMouse(CameraMouse):
#     def __init__(self, camera):
#         self.camera = camera
#         self.monitor = WindowsMonitor()
#         self.mouse = WindowsMouse()
#         self.gesture_recognition = KeyboardGestureRecognition()
#         self.prev_gray_img, self.color_img = self.camera.capture_frames()
#         tracker_params = dict( winSize  = (15, 15), maxLevel = 2,
#                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#         feature_params = dict(maxCorners = 500, qualityLevel = 0.2,
#                               minDistance = 2, blockSize = 7)
#         self.tracker = LukasKanadeResampling(tracker_params, feature_params, self.camera)


#     def run(self):
#         while True:
#             gray_img, color_img = self.camera.capture_frames()
#             self.tracker.update(gray_img, color_img)
#             # self.gesture_recognition.update()
#             self.execute_control()

#             ch = 0xFF & cv2.waitKey(1)
#             if ch == 27:
#                 break

class HandSegmentationMouse(CameraMouse):
    def __init__(self, camera, filter, filter_size, control, mouse, monitor):
        super().__init__(monitor, mouse, camera)
        # hard coded values for absolute tracking only
        self.x_ratio = self.monitor.width / 1000 #(self.camera.width)
        self.y_ratio = self.monitor.height / 500 #(self.camera.height)
        print("X Ratio {}, Y Ratio {}".format(self.x_ratio, self.y_ratio))

        self.gesture_recognition = KeyboardGestureRecognition()
        self.tracker = HandTracker(camera, filter_size, filter, alpha=0.7)
        self.lin_term = 1/100
        self.quad_term = 1/20000
        self.lin_sens = 7
        self.quad_sens = 4
        assert(control in ["vel", "abs",  "hybrid"])
        self.control = control
        self.last_gesture_update = 0

    def updateSens(self, _):
        self.lin_sens = cv2.getTrackbarPos("lin term", "FeedMe")
        self.quad_sens = cv2.getTrackbarPos("quad term", "FeedMe")

    def run(self):
        # i = 0
        flag = False
        cv2.namedWindow("FeedMe")
        cv2.createTrackbar("quad term", "FeedMe", \
          self.quad_sens, 10, self.updateSens)
        cv2.createTrackbar("lin term", "FeedMe", \
          self.lin_sens, 10, self.updateSens)
        while True:
            # grab frames
            gray_frame, color_frame = self.camera.capture_frames()

            if not self.tracker.found: # hand is lost
                if flag: # only execute once
                    self.tracker.handSeg.get_histogram()
                    flag = False
                self.tracker.global_recognition(color_frame)
            else: # found the hand lets track it
                if self.control == "vel":
                    self.tracker.get_velocity(color_frame)
                elif self.control == "abs":
                    self.tracker.get_position(color_frame)
                # Make sure we aren't doubling down on the same action
                if self.last_gesture_update < self.gesture_recognition.i:
                    self.last_gesture_update = self.gesture_recognition.i
                elif self.last_gesture_update == self.gesture_recognition.i and not (self.gesture_recognition.gesture == Gestures.drag):
                    self.gesture_recognition.gesture = Gestures.null

                self.execute_control()

                flag = True

            cv2.imshow("FeedMe", color_frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                cv2.destroyWindow("FeedMe")
                break

    def velocity_map(self):
        def get_gain(vel):
            return self.lin_term*self.lin_sens + self.quad_term*self.quad_sens*abs(vel)

        ret_x = int(self.tracker.vel_x * get_gain(self.tracker.vel_x))
        ret_y = int(self.tracker.vel_y * get_gain(self.tracker.vel_y))
        return ret_x, ret_y

    def position_map(self):
        def remap(x, y):
            # coordinate shift to new origin at (140, 100)
            x -= 140
            y -= 100
            x = np.clip(x, 0, 1000)
            y = np.clip(y, 0, 500)
            return x, y
        x_map, y_map = remap(self.tracker.pos_x, self.tracker.pos_y)
        x_cam = self.x_ratio * x_map
        y_cam = self.y_ratio * y_map
        return [int(self.monitor.width-x_cam), int(self.monitor.height-y_cam)] # accounts for camera facing down
