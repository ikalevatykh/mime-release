import numpy as np
import pybullet as pb

class GripperPositionController(object):
    def __init__(self, gripper, gain, max_force):
        joints = gripper.joints
        upper = np.max(np.abs(joints[0].info.limits))

        self._gripper = gripper
        self._joints = joints
        self._upper = upper
        self._gains = [gain, gain]
        self._forces = [max_force, max_force]
        self._target = gripper.width
        self._target_velocity = 0

    @property
    def max_forces(self):
        return self._forces

    @property
    def target_velocity(self):
        return self._target_velocity

    @target_velocity.setter
    def target_velocity(self, value):
        """ Command drives speed.
         Positive velocity - opening, otherwise closing"""
        self._target_velocity = float(value)

    def reset(self):
        self._target = self._gripper.width
        self._target_velocity = 0

    def step(self, dt):
        if self._target_velocity:
            self._step(dt, self._target_velocity)

    def _move(self, width):
        width = np.clip(width, 0, 1)
        pos = width * self._upper
        self._joints.control(
            controlMode=pb.POSITION_CONTROL, forces=self._forces,
            targetPositions=[-pos, pos], positionGains=[0.1, 0.1]
        )
        self._target = width

    def _step(self, dt, vel):
        width = self._target
        self._move(width + vel * dt)

    def grasped(self):
        return False

    def opened(self):
        return self._gripper.width > 0.99
