import numpy as np

from math import radians
from .controllable import Controllable


class ParallelGripper(Controllable):
    def __init__(self, body, left_tip_name, right_tip_name):
        super(ParallelGripper, self).__init__()

        # gripper fingertips
        tip_links = (body.link(left_tip_name), body.link(right_tip_name))

        # motor joints
        joint_ids = [
            next((i.index for i in reversed(tip.info.path) if i.is_prismatic), None)
            for tip in tip_links
        ]
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


class RG6Gripper(Controllable):
    def __init__(self, body, prefix="left_"):
        super(RG6Gripper, self).__init__()

        self._body = body

        self._grasping_frame = body.link(f"{prefix}gripper_body")

        self._joints = body.joints(
            [
                f"{prefix}gripper_joint",
                f"{prefix}gripper_finger_1_truss_arm_joint",
                f"{prefix}gripper_finger_1_safety_shield_joint",
                f"{prefix}gripper_finger_1_finger_tip_joint",
                f"{prefix}gripper_mirror_joint",
                f"{prefix}gripper_finger_2_truss_arm_joint",
                f"{prefix}gripper_finger_2_safety_shield_joint",
                f"{prefix}gripper_finger_2_finger_tip_joint",
            ]
        )

        self._open_default_joints_pos = [0.0] * 8
        self._close_default_joints_pos = [1.3] * 8

        self._right_tip = body.link(f"{prefix}gripper_finger_1_flex_finger")
        self._left_tip = body.link(f"{prefix}gripper_finger_2_flex_finger")

        self._max_limit, self._min_limit = 1.3, 0.0

        for jt in self.joints:
            jt.child_link.dynamics.contact_constraint = (100, 100)

    @property
    def joints(self):
        return self._joints

    def reset(self, close=False):
        if close:
            joints_pos = self._close_default_joints_pos
        else:
            joints_pos = self._open_default_joints_pos
        self._joints.reset(joints_pos)
        super(RG6Gripper, self).reset()

    def get_contacts(self):
        return self._body.get_contacts()
