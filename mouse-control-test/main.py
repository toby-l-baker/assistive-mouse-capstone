import cv2
import sys
import numpy as np
from mouse_states import OutOfRange
from win32api import GetSystemMetrics

print(cv2.__version__)

class Monitor():
    '''Class for storing data about the display'''
    def __init__(self):
        # Windows only at this stage
        self.dimensions = (GetSystemMetrics(0), GetSystemMetrics(1))
        print(self.dimensions)

class Camera():
    '''Class for storing camera parameters'''
    def __init__(self):
        # TODO
        break

class MouseFSM():
    '''Class for the mouse FSM, handles state changes and what code is executed'''
    def __init__(self):
        self.state = OutOfRange()

    def on_event(self, event):
        '''Handle a gesture input and use it to either remain in the current state
        or move states'''
        self.state =  self.state.on_event(event)

'''Get monitor information'''
monitor = Monitor()

'''Initialise state machine'''
state = MouseFSM()

'''TODO: Get camera information'''

# https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
# https://stackoverflow.com/questions/8593091/robust-hand-detection-via-computer-vision

while True:
    '''Check for keypresses to move between states'''
    if keyboard == 'i':
        state.on_event('in_range')
        print(state)
    elif keyboard == 'o':
        state.on_event('out_of_range')
        print(state)
    elif keyboard == 'd':
        state.on_event('drag')
        print(state)

    if state == 'OutOfRange':
        # Wait for state change
        break
    elif state == 'InRange':
        # Track hand motion
        # Calculate change in position
        # Move the mouse
        break
    elif state == 'Drag':
        # Track hand motion
        # Calculate change in position
        # Move the mouse and the item it has clicked on
        break
