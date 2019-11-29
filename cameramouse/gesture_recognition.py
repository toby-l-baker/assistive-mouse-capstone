"""
Author: Toby Baker
Title: Camera Mouse
Date Created: 28/NOV/2019
"""

import keyboard
from enum import Enum

class Gestures(Enum):
    null = 0
    click = 1
    double_click = 2
    right_click = 3
    out_of_range = 4


class GestureRecognition():

    def __init__(self):
        self.gesture = Gestures.null

    def update(self):
        pass

class KeyboardGestureRecognition(GestureRecognition):

    def __init__(self):
        super().__init__()
        self.pressed = {"a": False, "s": False, "d": False, "o": False} # a: click, s:d_click, d: r_click
        keyboard.on_press(self.update)

    def update(self, event):
        if event.name == "a":
            if self.pressed[event.name]:
                self.gesture = Gestures.null
                return
            # set state of key to pressed
            self.pressed[event.name] = True
            self.gesture = Gestures.click
        elif event.name == "s":
            if self.pressed[event.name]:
                self.gesture = Gestures.null
                return
            # set state of key to pressed
            self.pressed[event.name] = True
            self.gesture = Gestures.double_click
        elif event.name == "d":
            if self.pressed[event.name]:
                self.gesture = Gestures.null
                return
            # set state of key to pressed
            self.pressed[event.name] = True
            self.gesture = Gestures.right_click
        elif event.name == "o":
            if self.pressed[event.name]:
                self.gesture = Gestures.out_of_range
                return
            # set state of key to pressed
            self.pressed[event.name] = True
            self.gesture = Gestures.out_of_range
        else:
            self.pressed["a"] = False
            self.pressed["s"] = False
            self.pressed["d"] = False
            self.pressed["o"] = False
