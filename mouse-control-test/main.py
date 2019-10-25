#import cv2
import sys
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from mouse_states import OutOfRange
from pymouse import PyMouse
from time import sleep

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

def draw(vx, vy, dt):
    x_cur, y_cur = mouse.position()
    for i in range(0, len(vx)):
        x_cur = int(0.3*vx[i]) + x_cur
        y_cur = int(0.3*vy[i]) + y_cur
        mouse.move(x_cur, y_cur)
        sleep(dt)

'''Initialise our mouse class'''
mouse = PyMouse()

'''Get monitor information'''
monitor = Monitor()

'''Initialise state machine'''
state = MouseFSM()

'''TODO: Get camera information'''

# https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
# https://stackoverflow.com/questions/8593091/robust-hand-detection-via-computer-vision

# data = np.genfromtxt('mouse_data/circle.csv', delimiter=',')
# x = data[:, 0]
# y = data[:, 1]
# t = data[:, 3]
# vx = x/0.1
# vy = y/0.1
'''Setup variables for a preset mouse path'''
t = 10
dt = t / 100
theta = np.linspace(0, 6.2, t/dt)
r = 100
'''Calculate x, y, dx, dy for the desired shape'''
x = r*np.cos(theta) + np.random.normal(0, 5, len(theta)) #mu, sigma
y = r*np.sin(theta) + np.random.normal(0, 5, len(theta))
dx = x[1:] - x[:-1]
dy = y[1:] - y[:-1]
'''Calculate the velocity between points'''
vx = dx/dt
vy = dy/dt

plt.plot(x, y)
plt.show()

entry = 1

while True:
    '''Check for keypresses to move between states'''

    if str(state.state) == 'OutOfRange':
        # Wait for state change
        if entry:
            entry = 0
            pass
        else:
            if keyboard.is_pressed('i'):
                state.on_event('in_range')
                entry = 1
    elif str(state.state) == 'InRange':
        # Track hand motion
        # Calculate change in position
        # Move the mouse
        if entry:
            draw(vx, vy, dt)
            entry = 0
        else:
            if keyboard.is_pressed('o'):
                state.on_event('out_of_range')
                entry = 1
            elif keyboard.is_pressed('d'):
                state.on_event('drag')
                entry = 1

    elif str(state.state) == 'Drag':
        # Track hand motion
        # Calculate change in position
        # Move the mouse and the item it has clicked on
        if entry:
            entry = 0
            pass
        else:
            if keyboard.is_pressed('i'):
                state.on_event('in_range')
                entry = 1
