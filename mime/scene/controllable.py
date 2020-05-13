class Controllable(object):
    def __init__(self):
        self._controller = None

    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, value):
        self._controller = value

    def reset(self):
        if self._controller is not None:
            self._controller.reset()
