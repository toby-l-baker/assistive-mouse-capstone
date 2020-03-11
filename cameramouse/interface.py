"""
Author: Toby Baker
Title: Interface for dealing with various screen sizes and OS for mouse movement
Date Created: 28 Nov 2018
"""

import win32api, win32con # for windows mouse and monitor
import pyautogui


class Monitor():
    '''Class for storing data about the display'''
    def __init__(self):
        # Windows only at this stage
        pass

class WindowsMonitor(Monitor):
    def __init__(self):
        super().__init__()
        self.width, self.height = pyautogui.size()
        print('Mon Width: %d, Mon Height: %d' % (self.width, self.height))

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

    def drag(self):
        pass

    def move(self):
        pass


class WindowsMouse(Mouse):
    def __init__(self):
        super().__init__()
        self.state = "UP"

    def left_click(self, dx, dy):
        x, y = win32api.GetCursorPos()
        pyautogui.click(x=int(x+dx), y=int(y+dy), button='left')

    def right_click(self, dx, dy):
        x, y = win32api.GetCursorPos()
        pyautogui.click(x=int(x+dx), y=int(y+dy), button='right')

    def double_click(self, dx, dy):
        x, y = win32api.GetCursorPos()
        pyautogui.click(x=int(x+dx), y=int(y+dy), button='left', clicks=2)

    def mouse_down(self):
        x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)

    def mouse_up(self):
        x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)

    def moveD(self, dx, dy):
        x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,int(dx),int(dy),0,0)

    def move(self, dx, dy):
        x, y = win32api.GetCursorPos()
        win32api.SetCursorPos((int(dx+x), int(dy+y)))
