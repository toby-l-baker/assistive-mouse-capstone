"""
Author: Toby Baker
Title: Interface for dealing with various screen sizes and OS for mouse movement
Date Created: 28 Nov 2018
"""

# import pyautogui
import win32api, win32con, time # for windows mouse and monitor


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
        print('Mon Width: %d, Mon Height: %d' % (self.width, self.height))

class Mouse():
    '''Class for each mouse e.g. Winsdows/Linux/MacOS'''
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

    def left_click(self, x, y):
        self.mouse_down()
        self.mouse_up()

    def right_click(self, x, y):
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0,0,0)

    def double_click(self, x, y):
        self.mouse_down()
        self.mouse_up()
        self.mouse_down()
        self.mouse_up()

    def mouse_down(self):
        # x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)

    def mouse_up(self):
        # x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)

    def moveD(self, dx, dy): # MOVE DRAG
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,int(dx),int(dy),0,0)

    def move(self, x, y):
        win32api.SetCursorPos((int(x), int(y)))
