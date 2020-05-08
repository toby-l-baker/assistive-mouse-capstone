"""
utils.py
currently just has functions to load in classes
"""

"""
Author: Toby Baker
Title: loaders.py - functions to load hardware objects
Date Created: 28 Nov 2018
"""
import hardware.monitor as monitor
import hardware.mouse as mouse
import hardware.camera as camera

from control.controllers import *
from control.filters import *

from gesture_recognition.gestures import *
from gesture_recognition.keyboard import KeyboardGestureRecognition

from hand_tracking import tracking, colour_segmentation

class Loader():
    def load_monitor(os):
        if os not in ["linux", "windows"]:
            raise ValueError('Only Windows and Linux Supported')
        
        if os == "linux":
            print("[DEBUG] Loading Linux Monitor")
            return monitor.LinuxMonitor()
        elif os == "windows":
            print("[DEBUG] Loading Windows Monitor")
            return monitor.WindowsMonitor()

    def load_mouse(os):
        if os not in ["linux", "windows"]:
            raise ValueError('Only Windows and Linux Supported Currently')
        
        if os == "linux":
            print("[DEBUG] Loading Linux Mouse")
            return mouse.LinuxMouse()
        elif os == "windows":
            print("[DEBUG] Loading Windows Mouse")
            return mouse.WindowsMouse()

    def load_camera(opts):
        camera_type = opts["type"]
        src = int(opts["src"])

        if camera_type not in ["webcam", "realsense"]:
            raise ValueError('Only standard webcam or realsense camera supported currently')

        if camera_type == "realsense":
            print("[DEBUG] Loading Realsense Camera")
            return camera.RealSenseCamera()
        elif camera_type == "webcam":
            print("[DEBUG] Loading Webcam Camera")
            return camera.WebcamCamera(opts["src"])


    def load_control(opts, mouse, monitor, im_shape):
        def load_filter(filter_type, length):
            if filter_type == "iir":
                return IIRFilter(length)
            elif filter_type == "fir":
                return FIRFilter(length)
        filter_type = opts["filter"]
        filter_len = opts["filter_length"]
        control_type = opts["type"]

        if control_type not in ["absolute", "joystick", "relative"]:
            raise ValueError('Only absolute, relative or joystick style control supported')

        if filter_type not in ["iir", "fir"]:
            raise ValueError('Only iir and fir filters are supported')

        filter = load_filter(filter_type, filter_len)

        if control_type == "absolute":
            print("[DEBUG] Loading Absolute Controller")
            return AbsoluteControl(filter, mouse, monitor, im_shape)
        elif control_type == "relative":
            print("[DEBUG] Loading Relative Controller")
            return RelativeControl(filter, mouse)
        elif control_type == "joystick":
            print("[DEBUG] Loading Joystick Controller")
            return JoystickControl(filter, mouse, im_shape)

    def load_tracker(opts, camera):
        """ 
        Function for loading various hand trackers/segmentation methods.
        Currently there's one tracker and one segmentation method so it's not really implemented
        """
        seg_type = opts["segmentation"]
        track_type = opts["type"]
        if seg_type == "colour":
            return tracking.HandTracker(camera)

    def load_gesture(opts):
        gesture_type = opts["type"]
        if gesture_type not in ["keyboard", "none"]:
            raise ValueError('Select an available Gesture Recognition Module. Options: keyboard, none')
        
        if gesture_type == "keyboard":
            print("[DEBUG] Loading Keyboard Gesture Recognition")
            return KeyboardGestureRecognition()
        elif gesture_type == "none":
            print("[DEBUG] Loading Blank Gesture Recognition")
            return GestureRecognition()