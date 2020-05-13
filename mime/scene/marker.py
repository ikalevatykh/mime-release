class Marker(object):
    def __init__(self):
        self._body = None

    def make(self):
        raise NotImplementedError

    def update(self, marker):
        pass

    def show(self):
        if self._body is None:
            self._body = self.make()
        self.update(self._body)

    def hide(self):
        if self._body is not None:
            self._body.remove()
            self._body = None
