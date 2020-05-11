class State(object):
    """
    We define a state object which provides some utility functions for the
    individual states within the state machine.
    """

    def __init__(self):
        print(str(self))

    def on_event(self, event):
        """
        Handle events that are delegated to this State.
        """
        pass

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
        if event == 'in_range':
            return InRange()
        else:
            return self

class InRange(State):
    '''State where hand motion is tracked'''
    def on_event(self, event):
        if event == 'drag':
            return Drag()
        elif event == 'out_of_range':
            return OutOfRange()
        else:
            return self

class Drag(State):
    '''State where we have clicked and will move objects around on the screen
    or write in a note taking application'''
    def on_event(self, event):
        if event == 'in_range':
            return InRange()
        else:
            return self
