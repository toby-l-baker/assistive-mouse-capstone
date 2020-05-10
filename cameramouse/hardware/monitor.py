"""
Author: Toby Baker
Title: Interface for capturing metrics about monitors on different OS's
Date Created: 28 Nov 2018
"""

try:
    import win32api, win32con
except:
    import mouse
    import pyautogui

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
        print('Monitor Width: %d, Monitor Height: %d' % (self.width, self.height))
        print("[DEBUG] Windows Monitor Initialized")

class LinuxMonitor(Monitor):
    def __init__(self):
        super().__init__()
        self.width, self.height = pyautogui.size()
        print('Monitor Width: %d, Monitor Height: %d' % (self.width, self.height))
        print("[DEBUG] Linux Monitor Initialized")