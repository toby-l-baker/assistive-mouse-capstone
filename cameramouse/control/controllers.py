"""
Control Module

Contents:
Various cursor control methods - absolute, relative and joystick control
"""

import time, copy
import numpy as np
import gesture_recognition.gestures as gr


class Control():
    """
    Class used for mouse control, tackling the issue of how to map hand motion to cursor movements
    """
    def __init__(self, filter, mouse):
        self.filter = filter
        self.mouse = mouse

        # Dictionaries to hold filtered information about the velocity and position of the hand
        self.current = {"position": np.array([0, 0]), "velocity": np.array([0, 0]), "timestamp": time.time()}
        self.previous = {"position": np.array([0, 0]), "velocity": np.array([0, 0]), "timestamp": time.time()}

    def setup(self, point):
        """
        Setup needs to be done to fill the positions buffer
        """
        self.filter.positions[:, 0] *= point[0]
        self.filter.positions[:, 1] *= point[1]
        self.previous["timestamp"] = time.time()
        self.previous["position"] = point

    def map_absolute(self):
        """
        Returns the absolute position the cursor should move to 
        """
        raise NotImplementedError
    
    def map_relative(self):
        """
        Return how much the cursor should move
        """
        raise NotImplementedError
    
    def update(self, point):
        """
        When a new postion of the hand is found we need to add that position to the filter
        Inputs:
            point: a (2, ) numpy.ndarray holding the (x, y) coordinates of the hand in the image frame
        """
        self.current["timestamp"] = time.time()
        self.current["position"] = self.filter.get_filtered_position(point)
        dt = self.current["timestamp"] - self.previous["timestamp"]
        self.current["velocity"] = -(self.current["position"] - self.previous["position"]) / dt
        self.previous = copy.copy(self.current)
    
    def execute(self, gesture):
        """
        According to the gesture being made this function executes cursor movements and actions
        """
        if gesture == gr.Gestures.out_of_range:
            return
        else:
            x, y = self.map_absolute()
            dx, dy = self.map_relative()
            
            # Execution of gestures
            if gesture == gr.Gestures.drag:
                if self.mouse.state == "UP":
                    self.mouse.mouse_down()
                    self.mouse.state = "DOWN"
                else:
                    self.mouse.moveD(dx, dy) # needs to use differences
            else:
                if self.mouse.state == "DOWN":
                    self.mouse.mouse_up()
                    self.mouse.state = "UP"
                if gesture == gr.Gestures.click:
                    self.mouse.left_click()
                elif gesture == gr.Gestures.double_click:
                    self.mouse.double_click()
                elif gesture == gr.Gestures.right_click:
                    self.mouse.right_click()
                else:
                    self.mouse.move(x, y)
                    # print("Moving to {}".format((x, y)))

class AbsoluteControl(Control):
    def __init__(self, filter, mouse, monitor, im_shape):
        super().__init__(filter, mouse)
        self.monitor = monitor

        # shrink useful area since hand centroid never reaches the edge of the frame
        self.new_width = int(0.5 * im_shape[0])
        self.new_height = int(0.5 * im_shape[1])
        # print("Abs New Region {}".format((self.new_width, self.new_height)))

        # calculate the new origin we will be using when mapping abs position to the cursor position
        self.new_orig_x = (im_shape[0] - self.new_width) // 2
        self.new_orig_y = (im_shape[1] - self.new_height) // 2    
        # print("Abs New Origin {}".format((self.new_orig_x, self.new_orig_y)))

        # scaling ratios
        self.x_ratio = self.monitor.width**2 / (self.new_width * im_shape[0]) 
        self.y_ratio = self.monitor.height**2 / (self.new_height * im_shape[1])
        # print("RATIOs {}".format((self.x_ratio, self.y_ratio)))

        print("[DEBUG] Absolute Controller Initialised")

    def remap(self, point):
        # coordinate shift to new origin
        x, y = point
        x -= self.new_orig_x
        y -= self.new_orig_y
        x = np.clip(x, 0, self.new_width)
        y = np.clip(y, 0, self.new_height)
        return np.array([x, y])

    def map_absolute(self):
        pos_mapped = self.remap(self.current["position"])
        x_cam = self.x_ratio * pos_mapped[0]
        y_cam = self.y_ratio * pos_mapped[1]
        # print("Remapped Position {}".format((x_cam, y_cam)))
        return [int(self.monitor.width-x_cam), int(self.monitor.height-y_cam)] # accounts for camera facing down
    
    def map_relative(self):
        pos_mapped = self.remap(self.current["position"])
        x_cam = self.x_ratio * pos_mapped[0]
        y_cam = self.y_ratio * pos_mapped[1]
        print("Screen Position {}".format((int(self.monitor.width-x_cam), int(self.monitor.height-y_cam))))
        
        cx, cy = self.mouse.position()
        dx = int(self.monitor.width-x_cam) - cx
        dy = int(self.monitor.height-y_cam) - cy
        return [dx, dy] 

class RelativeControl(Control):
    def __init__(self, filter, mouse):
        super().__init__(filter, mouse)
        self.lin_term = 7/100
        self.quad_term = 4/20000
        print("[DEBUG] Relative Controller Initialised")

    def map_absolute(self):
        """
        Returns the x, y for the cursor to move to
        """
        vel = self.current["velocity"]
        ret_x = int(vel[0] * self.get_gain(vel[0]))
        ret_y = int(vel[1] * self.get_gain(vel[1]))
        cx, cy = self.mouse.position()
        return np.array([ret_x+cx, ret_y+cy])

    def map_relative(self):
        """
        Returns the dx, dy for the cursor to move 
        """

        vel = self.current["velocity"]
        ret_x = int(vel[0] * self.get_gain(vel[0]))
        ret_y = int(vel[1] * self.get_gain(vel[1]))
        return np.array([ret_x, ret_y])

    def get_gain(self, vel):
        return self.lin_term + self.quad_term*abs(vel)

class JoystickControl(Control):
    def __init__(self, filter, mouse, im_shape):
        super().__init__(filter, mouse)
        self.centre = [im_shape[0] // 2, im_shape[1] // 2]
        self.centre_radius = 85
        self.scale = 5 / self.centre_radius
        print("[DEBUG] Joystick Controller Initialised")

    def get_vector_from_centre(self, point): 
        vec = [point[0] - self.centre[0], point[1] - self.centre[1]]
        return np.array(vec)

    def map_absolute(self):
        vector = self.get_vector_from_centre(self.current["position"])
        change = None
        if np.linalg.norm(vector) < self.centre_radius:
            change = 0.0 * vector
        else:
            change = self.scale * vector
        cx, cy = self.mouse.position()
        return np.array([change[0] + cx, change[1] + cy])
    
    def map_relative(self):
        vector = self.get_vector_from_centre(self.current["position"])
        if np.linalg.norm(vector) < self.centre_radius:
            return 0.0 * vector
        else:
            return self.scale * vector