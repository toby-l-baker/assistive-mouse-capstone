"""
Author: Toby Baker
Title: Interface for dealing with various screen sizes and OS for mouse movement
Date Created: 28 Nov 2018
"""

import win32api, win32con # for windows mouse
from mouse_states import OutOfRange


class Monitor():
    '''Class for storing data about the display'''
    def __init__(self):
        # Windows only at this stage
        pass

class WindowsMonitor(Monitor):
    def __init__(self):
        super().__init__()
        self.width = win32api.GetSystemMetrics(0)
        self.height = win32api.GetSystemMetrics(1)

class Mouse():
    '''Class for each mouse e.g. Windows/Linux/MacOS'''
    def __init__(self):
        pass

    def left_click(self):
        pass

    def right_click(self):
        pass

    def double_click(self):
        pass

    def move(self):
        pass


class WindowsMouse(Mouse):
    def __init__(self):
        super().__init__()

    def left_click(self):
        x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

    def right_click(self):
        x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)

    def double_click(self):
        x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

    def move(self, x_m, y_m):
        x, y = win32api.GetCursorPos()
        win32api.SetCursorPos((int(x_m+x), int(y_m+y)))
