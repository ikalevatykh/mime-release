import numpy as np
from .controllable import Controllable


class ParallelGripper(Controllable):
    def __init__(self, body, left_tip_name, right_tip_name):
        super(ParallelGripper, self).__init__()

        # gripper fingertips
        tip_links = (body.link(left_tip_name), body.link(right_tip_name))

        # motor joints
        joint_ids = [next((i.index for i in reversed(tip.info.path) if i.is_prismatic), None)
                     for tip in tip_links]
        joints = body[joint_ids]
        upper = np.max(np.abs(joints[0].info.limits))

        # reset at open position
        joints.reset([-upper, upper])

        self._joints = joints
        self._upper = upper
        self._controller = None

    def reset(self, width):
        pos = width * self._upper
        self._joints.reset([-pos, pos])
        super(ParallelGripper, self).reset()

    @property
    def joints(self):
        return self._joints

    @property
    def width(self):
        positions = self._joints.state.positions
        return np.sum(np.abs(positions)) / 2 / self._upper

    @property
    def reaction_forces(self):
        t1, t2 = self._joints.state.applied_joint_motor_torques
        return t1, -t2

