#import cv2
import sys
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from mouse_states import OutOfRange
from pynput.mouse import Button, Controller

class Monitor:
    '''Class for storing data about the display'''
    def __init__(self):
        # Windows only at this stage
        print("NO")

class Camera:
    '''Class for storing camera parameters'''
    def __init__(self):
        # TODO
        pass

class MouseFSM(object):
    '''Class for the mouse FSM, handles state changes and what code is executed'''
    def __init__(self):
        self.state = OutOfRange()

    def on_event(self, event):
        '''Handle a gesture input and use it to either remain in the current state
        or move states'''
        self.state =  self.state.on_event(event)

'''Initialise our mouse class'''
mouse = Controller()

'''Get monitor information'''
monitor = Monitor()

'''Initialise state machine'''
state = MouseFSM()

'''TODO: Get camera information'''

# https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
# https://stackoverflow.com/questions/8593091/robust-hand-detection-via-computer-vision

data = np.genfromtxt('mouse_data/circle.csv', delimiter=',')
x = data[:, 0]
y = data[:, 1]
t = data[:, 3]

vx = x/0.1
vy = y/0.1
plt.plot(t[1:], vx[1:])
#plt.show()

while True:
    '''Check for keypresses to move between states'''
    if keyboard.is_pressed('i'):
        state.on_event('in_range')
    elif keyboard.is_pressed('o'):
        state.on_event('out_of_range')
    elif keyboard.is_pressed('d'):
        state.on_event('drag')

    if state.state == 'OutOfRange':
        # Wait for state change
        pass
    elif state.state == 'InRange':
        # Track hand motion
        # Calculate change in position
        # Move the mouse
        for i in range(1, len(x)):
            mouse.move(vx[i], vy[i])
        pass
    elif state.state == 'Drag':
        # Track hand motion
        # Calculate change in position
        # Move the mouse and the item it has clicked on
        pass
