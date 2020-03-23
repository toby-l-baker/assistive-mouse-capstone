"""
Author: Toby Baker
Title: Camera Mouse
Date Created: 28/NOV/2019
"""

import keyboard, time
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

    def update(self):
        pass

class KeyboardGestureRecognition(GestureRecognition):

    def __init__(self):
        super().__init__()
        self.pressed = {"a": False, "s": False, "d": False, "o": False} # a: click, s:d_click, d: r_click
        keyboard.hook(self.update)
        # self.i = 0

    def update(self, event):
        #print(event.event_type)
        self.i += 1
        if event.name == "s":
            if event.event_type == "down":
                if self.pressed[event.name]:
                    self.gesture = Gestures.null # debouncing
                    return
                # set state of key to pressed
                self.pressed[event.name] = True
                self.gesture = Gestures.click
            else:
                self.pressed[event.name] = False
                self.gesture = Gestures.null
        elif event.name == "a":
            if event.event_type == "down":
                if self.pressed[event.name]:
                    self.gesture = Gestures.null
                    return
                # set state of key to pressed
                self.pressed[event.name] = True
                self.gesture = Gestures.double_click
            else:
                self.pressed[event.name] = False
                self.gesture = Gestures.null
        elif event.name == "r":
            if event.event_type == "down":
                if self.pressed[event.name]:
                    self.gesture = Gestures.null
                    return
                # set state of key to pressed
                self.pressed[event.name] = True
                self.gesture = Gestures.right_click
            else:
                self.pressed[event.name] = False
                self.gesture = Gestures.null
        elif event.name == "d":
            if event.event_type == "down":
                if self.pressed[event.name]:
                    return
                # set state of key to pressed
                self.pressed[event.name] = True
                if self.gesture == Gestures.drag:
                    self.gesture = Gestures.null
                else:
                    self.gesture = Gestures.drag
            else:
                self.pressed[event.name] = False
        elif event.name == "o":
            if event.event_type == "down":
                # set state of key to pressed
                self.gesture = Gestures.out_of_range
            else:
                self.pressed[event.name] = False
                self.gesture = Gestures.null
