import enum

import numpy as np
import pybullet as pb

from pyquaternion import Quaternion


class Command(enum.Enum):
    Standby = 0
    JointsVelocity = 1
    ToolVelocity = 2
    JointsPosition = 3
    ToolPosition = 4


class ArmPositionController:
    def __init__(self, arm, gains):
        """
        Constructor.
        Args:
            arm - Arm object.
            gains (vec, arm DOF) - joints P-controller gains
        """

        max_forces = [i.max_force for i in arm.joints.info]
        if np.isscalar(gains):
            gains = [gains] * len(arm.joints)
        self._arm = arm
        self._max_forces = max_forces
        self._gains = gains
        self._command = Command.Standby
        self._workspace = None
        self._orientation_constraints = None
        self.reset()

    # Workspace

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, value):
        self._workspace = value

    # Constraints

    @property
    def orientation_constraints(self):
        return self._orientation_constraints

    @workspace.setter
    def orientation_constraints(self, value):
        self._orientation_constraints = value

    # State

    @property
    def joints_target(self):
        return self._joints_target_position

    @property
    def joints_error(self):
        return np.subtract(self._joints_target_position, self._arm.joints_position)

    @property
    def joints_target_velocity(self):
        return self._joints_target_velocity

    @property
    def tool_target(self):
        return self._tool_target_position

    @property
    def tool_target_velocity(self):
        return self._tool_target_velocity

    # Commands

    @joints_target_velocity.setter
    def joints_target_velocity(self, value):
        """ Command target joints speed """
        self._joints_target_velocity = value
        self._command = Command.JointsVelocity

    @tool_target_velocity.setter
    def tool_target_velocity(self, value):
        """ Command target tool velocity """
        self._tool_target_velocity = value
        self._command = Command.ToolVelocity

    @joints_target.setter
    def joints_target(self, value):
        """ Command target joint position """
        self._joints_target_position = value
        self._command = Command.JointsPosition

    @tool_target.setter
    def tool_target(self, value):
        """ Command target tool position """
        cpos, corn = self._tool_target_position
        pos, orn = value
        if pos is not None:
            cpos = pos
        if orn is not None:
            corn = orn

        self._tool_target_position = cpos, corn
        self._command = Command.ToolPosition

    # Control

    def reset(self):
        """ Called at start / after arm position reset """
        self._joints_target_position = self._arm.joints.state.positions
        self._joints_target_velocity = (0,) * len(self._joints_target_position)
        self._tool_target_position = self._arm.tool.state.position
        self._tool_target_velocity = (0, 0, 0), (0, 0, 0)
        self._joints_move(1, self._joints_target_position)

    def step(self, dt):
        """ Called each simulation step """
        if Command.JointsVelocity == self._command:
            dq = self._joints_target_velocity
            self._joints_step(dt, dq)
            self._tool_target_position = self._arm.kinematics.forward(
                self._joints_target_position
            )
        elif Command.ToolVelocity == self._command:
            v, w = self._tool_target_velocity
            self._tool_step(dt, v, w)
        elif Command.JointsPosition == self._command:
            q = self._joints_target_position
            self._joints_move(dt, q)
        elif Command.ToolPosition == self._command:
            p, o = self._tool_target_position
            self._tool_move(dt, p, o)

    def _joints_move(self, dt, joints_pos):
        joints_vel = np.subtract(joints_pos, self._joints_target_position) / dt
        self._arm.joints.control(
            controlMode=pb.POSITION_CONTROL,
            forces=self._max_forces,
            targetPositions=joints_pos,
            positionGains=self._gains,
        )
        self._joints_target_position = joints_pos
        self._joints_target_velocity = joints_vel

    def _joints_step(self, dt, dq):
        joints_pos = self._joints_target_position
        self._joints_move(dt, joints_pos + np.array(dq) * dt)

    def _tool_move(self, dt, pos, orn):
        if self._check_workspace(pos, orn):
            q_init = self._arm.joints.state.positions
            q_solv = self._arm.kinematics.inverse(pos, orn, q_init)
            if q_solv is None:
                # print('Cannot find IK solution')
                return
            if np.max(np.abs(np.subtract(q_solv, q_init))) > 0.2:
                # print('IK solution discontinuity')
                return
            self._joints_move(dt, q_solv)
            self._tool_target_position = (pos, orn)

    def _tool_step(self, dt, lin_vel, ang_vel):
        pos, orn = self._tool_target_position
        if lin_vel is not None:
            pos = pos + np.array(lin_vel) * dt
        if ang_vel is not None:
            dorn = np.hstack([0.5 * np.array(ang_vel) * dt, 1.0])
            _, orn = pb.multiplyTransforms((0, 0, 0), dorn, (0, 0, 0), orn)
            orn = np.array(orn) / np.linalg.norm(orn)
        self._tool_move(dt, pos, orn)

    def _check_workspace(self, pos, orn):
        if self._workspace is not None:
            low, high = self.workspace
            if np.any((pos < low) | (pos > high)):
                return False

        if self._orientation_constraints is not None:
            orn0, axis, low, high = self._orientation_constraints

            q0 = Quaternion(orn0[3], *orn0[:3])
            q1 = Quaternion(orn[3], *orn[:3])
            dq = q1.inverse * q0

            k = np.dot(axis, dq.axis)
            a = dq.angle * np.sign(k)

            if a < low or a > high:
                return False

        return True
