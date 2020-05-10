"""Parent Classes for Gesture Recognition and Gesture enum"""

from enum import Enum

class Gestures(Enum):
    null = 0
    click = 1
    double_click = 2
    right_click = 3
    drag = 4
    out_of_range = 5


class GestureRecognition():

    def __init__(self):
        self.gesture = Gestures.null
        self.i = 0
        print("[DEBUG] Empty Gesture Input Initialized")


    def update(self):
        pass