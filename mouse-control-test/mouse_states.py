from states import State

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
