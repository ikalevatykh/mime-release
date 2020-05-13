from enum import Enum
from math import radians

import numpy as np
import pybullet as pb

from .constraint import Constraint
from .controllable import Controllable

from .joint import JointArray


class GraspMode(Enum):
    Basic = 0
    Pinch = 1
    Wide = 2
    Scissor = 3


class GraspState(Enum):
    Opened = 0
    Closed = 1
    Opening = 2
    Closing = 3


class RobotiqGripper(Controllable):
    def __init__(self, body):
        super(RobotiqGripper, self).__init__()
        controller = RobotiqGripperController(self)

        palm = body.link('palm')
        joints = body.joints([
            'palm_finger_1_joint',
            'palm_finger_2_joint',
            'finger_1_joint_1',
            'finger_1_joint_2',
            'finger_1_joint_3',
            'finger_2_joint_1',
            'finger_2_joint_2',
            'finger_2_joint_3',
            'finger_middle_joint_1',
            'finger_middle_joint_2',
            'finger_middle_joint_3',
        ])

        palm_lim = (radians(-10), radians(15))
        jnt1_lim = (radians(0), radians(70))
        jnt2_lim = (radians(0), radians(90))
        jnt3_lim = (radians(-55), radians(43))
        limits = (palm_lim, ) * 2 + (jnt1_lim, jnt2_lim, jnt3_lim) * 3

        self._body = body
        self._palm = palm
        self._joints = joints
        self._limits = limits
        self._controller = controller

    def reset(self, mode, finger_pos=[0, 0, 0], state=GraspState.Opened):
        if isinstance(mode, str):
            mode = GraspMode[mode]
        if isinstance(state, str):
            state = GraspState[state]
        palm_pos = 0.0
        if mode == GraspMode.Pinch:
            palm_pos = -0.15
        elif mode == GraspMode.Wide:
            palm_pos = 0.25
        position = [palm_pos, -palm_pos] + finger_pos * 3
        self._joints.reset(position)
        self.controller.reset(mode, position, state)

    @property
    def palm(self):
        return self._palm

    @property
    def joints(self) -> JointArray:
        return self._joints

    @property
    def reaction_forces(self):
        return [0, 0]

    @property
    def width(self):
        return 0

    @property
    def limits(self):
        return self._limits

    def get_contacts(self):
        return self._body.get_contacts()


class RobotiqGripperController(object):
    def __init__(self, gripper):
        self.manual_control = False
        self._gripper = gripper
        self._state = GraspState.Opened
        self._gains = np.array((2.0, ) * 2 + (1.0, 1.0, 0.8) * 3)
        self._force = np.array((10., ) * 2 + (12.4, ) * 6 + (25., ) * 3)
        self._blocked_moment = np.array((0.1, ) * 6 + (0.2, ) * 3)
        self._kinematic_grasp = True
        self._kinematic_constraint = None
        self._object_grasped = False
        self._object_attached = None

    def reset(self, mode, position, state):
        assert mode != GraspMode.Scissor, 'Scissor mode not supported yet'
        self._grasp_mode = mode
        self._state = state
        self._motor_position = [0, 0, 0]
        self._target_position = position
        self._target_velocity = 0  # drive velocity
        self._phalanx_blocked = [
            0,
        ] * 9  # flag 0, 1
        self._phalanx_limited = [
            0,
        ] * 9  # flag -1, 0, 1
        self._phalanx_contacts = [
            [],
        ] * 9
        self._history_position = []
        self._history_state = []
        self._detach_grasped_body()
        self._move_to(position)

    @property
    def target_velocity(self):
        return self._target_velocity

    @target_velocity.setter
    def target_velocity(self, value):
        """ Command drives speed.
         Positive velocity - opening, otherwise closing """
        if value > 0.0:
            if self._state != GraspState.Opened:
                self._state = GraspState.Opening
        elif value < 0.0:
            if self._state != GraspState.Closed:
                self._state = GraspState.Closing
        self._target_velocity = float(value)

    @property
    def state(self):
        if self._state == GraspState.Opened:
            return 2
        elif self._state == GraspState.Closed:
            return -2
        else:
            return self._target_velocity

    @property
    def kinematic_grasp(self):
        return self._kinematic_grasp

    @kinematic_grasp.setter
    def kinematic_grasp(self, value):
        """ If a kinematic grasp flag is set an object fixing
        in a gripper after closing. """
        self._kinematic_grasp = value

    @property
    def motor_position(self):
        ''' Motor positions for all three fingers  '''
        return self._motor_position

    def adjust_max_force(self, coeff):
        self._force *= coeff
        self._blocked_moment *= coeff

    def step(self, dt):
        """ Called each simulation step """
        position = self._target_position.copy()
        blocked = self._phalanx_blocked
        limited = self._phalanx_limited

        if self.manual_control:
            return

        if GraspState.Opening == self._state:
            if self._kinematic_grasp:
                self._detach_grasped_body()
            if self._history_position:
                # little hack: opening is a rollback of closing
                position = self._history_position.pop()
                blocked[:], limited[:] = self._history_state.pop()
            else:
                self._state = GraspState.Opened
        elif GraspState.Closing == self._state:
            self._history_position.append(position.copy())
            self._history_state.append((blocked.copy(), limited.copy()))
            self._update_limited()
            self._update_blocked()
            step = self._compute_step(dt, blocked, limited)
            if any(step):
                position[2:] = [p + s for p, s in zip(position[2:], step)]
            else:
                if self._kinematic_grasp:
                    self._attach_grasped_body()
                    contacted_links = set(
                        [c for cts in self._phalanx_contacts for c in cts])
                    self._object_grasped = len(contacted_links) > 0
                self._state = GraspState.Closed

        if position != self._target_position:
            self._move_to(position)

    def grasped(self):
        return GraspState.Closed == self._state and self._object_grasped

    def opened(self):
        return GraspState.Opened == self._state

    def _move_to(self, position):
        self._gripper.joints.control(
            controlMode=pb.POSITION_CONTROL,
            targetPositions=position,
            positionGains=self._gains,
            forces=self._force)
        self._target_position = position

    def _update_limited(self):
        limits = self._gripper.limits
        position = self._target_position
        self._phalanx_limited = [
            -1 * (p < l) + 1 * (p > u) for p, (l, u) in zip(position, limits)
        ]

    def _update_blocked(self):
        contacts = self._gripper.get_contacts()
        for i in range(9):  # for each phalanx
            if self._phalanx_blocked[i]:
                continue

            body = self._gripper._body
            joint = self._gripper.joints[2 + i]
            link = joint.child_link
            p0, o0 = link.state.world_link_frame_position
            m0 = pb.getMatrixFromQuaternion(o0)
            _, _, z = np.array(m0).reshape((3, 3)).T

            link_contacts = [
                c for c in contacts if c.link_a == link and c.body_b != body
            ]
            if link_contacts:
                f = [
                    np.multiply(c.contact_normal_on_b, c.normal_force)
                    for c in link_contacts
                ]
                p = [c.position_on_a for c in link_contacts]
                moment = np.sum(np.abs(np.cross(f, np.subtract(p, p0))))
                if moment > self._blocked_moment[i]:
                    links = set([c.link_b for c in link_contacts])
                    self._phalanx_contacts[i] = links
                    self._phalanx_blocked[i] = 1
            else:
                self._phalanx_contacts[i] = []
                self._phalanx_blocked[i] = 0

    def _compute_step(self, dt, blocked, limited):
        drive_step = self._target_velocity * dt
        f1 = -0.192 * drive_step
        f2 = -0.384 * drive_step
        f3 = -0.384 * drive_step

        step = [
            0,
        ] * 9
        for i in range(0, 9, 3):  # for each finger
            state = (blocked[i:i + 3], limited[i:i + 3])
            if state == ([0, 0, 0], [0, 0, 0]):
                step[i:i + 3] = (f1, 0, -f1)  # 1
            elif state == ([0, 0, 0], [0, 0, -1]):
                step[i:i + 3] = (f1, 0, 0)  # 1'
            elif state in [([1, 0, 0], [0, 0, 0]), ([0, 0, 0], [1, 0, 0])]:
                step[i:i + 3] = (0, f2, -f2)  # 2
            elif state in [([1, 0, 0], [0, 0, -1]), ([0, 0, 0], [1, 0, -1])]:
                step[i:i + 3] = (0, f2, 0)  # 2'
            elif (state[0][1:], state[1][1:]) in [([1, 0], [0, 0]),
                                                  ([0, 0], [1, 0])]:
                step[i:i + 3] = (0, 0, f3)  # 3'
            elif state == ([1, 1, 0], [0, 0, -1]):
                step[i:i + 3] = (0, 0, f3)  # 3'
            if any(step[i:i + 3]):
                self._motor_position[int(i / 3)] -= drive_step

        return step

    def _attach_grasped_body(self):
        palm = self._gripper.palm
        contacted_links = set(
            [c for cts in self._phalanx_contacts for c in cts])
        if contacted_links:
            link = contacted_links.pop()
            self._kinematic_constraint = Constraint.create_fixed(palm, link)
            for jt in self._gripper.joints:
                jt.child_link.dynamics.contact_constraint = (0.1, 0.1)

    def _detach_grasped_body(self):
        if self._kinematic_constraint is not None:
            self._kinematic_constraint.remove()
            self._kinematic_constraint = None
            for jt in self._gripper.joints:
                jt.child_link.dynamics.contact_constraint = (100, 100)

    def attach_object(self, link):
        palm = self._gripper.palm
        self._object_attached = link
        self._kinematic_constraint = Constraint.create_fixed(palm, link)
        for jt in self._gripper.joints:
            jt.child_link.dynamics.contact_constraint = (0.1, 0.1)

    def detach_object(self):
        detached_object = None
        if self._kinematic_constraint is not None:
            self._kinematic_constraint.remove()
            self._kinematic_constraint = None
            detached_object = self._object_attached
            self._object_attached = None
            for jt in self._gripper.joints:
                jt.child_link.dynamics.contact_constraint = (100, 100)
        return detached_object
