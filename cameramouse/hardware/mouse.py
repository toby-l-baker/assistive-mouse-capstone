"""
Author: Toby Baker
Title: Mouse interface to abstract different mouse APIs
Date Created: 28 Nov 2018
"""

try:
    import win32api, win32con
    import mouse
except:
    import mouse
    import pyautogui

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
        print("[DEBUG] Windows Mouse Initialized")

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
        self.state = "DOWN"

    def mouse_up(self):
        # x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
        self.state = "UP"

    def moveD(self, dx, dy): # MOVE DRAG
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,int(dx),int(dy),0,0)

    def move(self, x, y):
        win32api.SetCursorPos((int(x), int(y)))

    def position(self):
        return win32api.GetCursorPos()

class LinuxMouse(Mouse):
    def __init__(self):
        super().__init__()
        print("[DEBUG] Linux Mouse Initialized")


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

