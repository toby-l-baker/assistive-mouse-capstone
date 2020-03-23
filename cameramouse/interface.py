"""
Author: Toby Baker
Title: Interface for dealing with various screen sizes and OS for mouse movement
Date Created: 28 Nov 2018
"""

# import pyautogui
import time # for windows mouse and monitor
try:
    import win32api, win32con
except:
    import mouse
    import pyautogui
    print("cannot import win32api or win32con")


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

class LinuxMonitor(Monitor):
    def __init__(self):
        super().__init__()
        self.width, self.height = pyautogui.size()
        print('Mon Width: %d, Mon Height: %d' % (self.width, self.height))

class Mouse():
    '''Class for each mouse e.g. Winsdows/Linux/MacOS'''
    def __init__(self):
        self.state = "UP"
        pass

    def left_click(self):
        pass

    def right_click(self):
        pass

    def double_click(self):
        pass

    def move(self, x, y):
        pass

    def moveD(self, dx, dy):
        pass

    def position(self):
        return 0, 0


class WindowsMouse(Mouse):
    def __init__(self):
        super().__init__()

    def left_click(self):
        self.mouse_down()
        self.mouse_up()

    def right_click(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0,0,0)

    def double_click(self):
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

    def position(self):
        return win32api.GetCursorPos()

class LinuxMouse(Mouse):
    def __init__(self):
        super().__init__()

    def left_click(self):
        self.mouse_down()
        self.mouse_up()

    def right_click(self):
        mouse.press(button='right')
        mouse.release(button='right')

    def double_click(self):
        self.mouse_down()
        self.mouse_up()
        self.mouse_down()
        self.mouse_up()

    def mouse_down(self):
        mouse.press(button='left')

    def mouse_up(self):
        mouse.release(button='left')

    def moveD(self, dx, dy): # MOVE DRAG
        mouse.move(dx, dy, absolute=False, duration=0)

    def move(self, x, y):
        mouse.move(x, y, absolute=True, duration=0)

    def position(self):
        return mouse.get_position()