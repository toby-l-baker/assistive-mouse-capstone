from gesture_recognition.gestures import Gestures

class State(object):
    """
    We define a state object which provides some utility functions for the
    individual states within the state machine.
    """

    def __init__(self, mouse):
        print(str(self))
        self.mouse = mouse

    def on_event(self, event):
        """
        Handle events that are delegated to this State.
        """
        pass

    def on_entry(self, event):
        """
        What to do when you enter a given state
        """
        return

    def __repr__(self):
        """
        Leverages the __str__ method to describe the State.
        """
        return self.__str__()

    def __str__(self):
        """
        Returns the name of the State.
        """
        return self.__class__.__name__


class OutOfRange(State):
    '''State when the hand is out of range.'''
    def on_event(self, event):
        if event != Gestures.out_of_range:
            new_state = InRange(self.mouse)
            new_state.on_entry()
            return new_state
        else:
            return self

    def on_entry(self):
        if self.mouse.state == "DOWN":
            self.mouse.mouse_up()
        return   

    def execute(self, gesture, cmd_x, cmd_y):
        return 

class InRange(State):
    '''State where hand motion is tracked'''
    def on_event(self, event):
        if event == Gestures.drag:
            new_state = Drag(self.mouse)
            new_state.on_entry()
            return new_state
        elif event == Gestures.out_of_range:
            new_state = OutOfRange(self.mouse)
            new_state.on_entry()
            return new_state
        else:
            return self

    def on_entry(self):
        if self.mouse.state == "DOWN":
            self.mouse.mouse_up()
        return     

    def execute(self, gesture, cmd_x, cmd_y):
        if gesture == Gestures.click:
            self.mouse.left_click()
        elif gesture == Gestures.double_click:
            self.mouse.double_click()
        elif gesture == Gestures.right_click:
            self.mouse.right_click()
        self.mouse.move(cmd_x, cmd_y)
        return

class Drag(State):
    '''State where we have clicked and will move objects around on the screen
    or write in a note taking application'''
    def on_event(self, event):
        if event != Gestures.drag:
            new_state = InRange(self.mouse)
            new_state.on_entry()
            return new_state
        else:
            return self

    def on_entry(self):
        self.mouse.mouse_down()
        return 

    def execute(self, gesture, cmd_x, cmd_y):
        self.mouse.moveD(cmd_x, cmd_y) # needs to use differences
        return