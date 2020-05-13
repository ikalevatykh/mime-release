import pybullet as pb


class VR(object):
    MenuButton = 1
    GripButton = 2
    ThumbButton = 32
    TriggerButton = 33

    @staticmethod
    def events():
        return [VREvent(e) for e in pb.getVREvents()]


class VREvent(object):
    def __init__(self, event):
        self._event = event

    @property
    def controller_id(self):
        return self._event[0]

    @property
    def position(self):
        return self._event[1], self._event[2]

    @property
    def analog(self):
        return self._event[3]

    @property
    def buttons(self):
        return self._event[6]

    def button_is_down(self, id):
        return self.buttons[id] == pb.VR_BUTTON_IS_DOWN

    def button_was_triggered(self, id):
        return self.buttons[id] & pb.VR_BUTTON_WAS_TRIGGERED

    def button_was_released(self, id):
        return self.buttons[id] & pb.VR_BUTTON_WAS_RELEASED

    @property
    def device_type(self):
        return self._event[7]

    @property
    def device_is_controller(self):
        return self.device_type == pb.VR_DEVICE_CONTROLLER

    @property
    def device_is_generic_tracker(self):
        return self.device_type == pb.VR_DEVICE_GENERIC_TRACKER
