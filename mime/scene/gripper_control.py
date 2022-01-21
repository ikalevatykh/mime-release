from enum import Enum
import numpy as np
import pybullet as pb

from .constraint import Constraint


class GraspState(Enum):
    Opened = 0
    Closed = 1
    Opening = 2
    Closing = 3


class GripperPositionController(object):
    def __init__(self, gripper, gain, max_force):
        joints = gripper.joints
        upper = np.max(np.abs(joints[0].info.limits))

        self._state = GraspState.Opened
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
        """Command drives speed.
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
            controlMode=pb.POSITION_CONTROL,
            forces=self._forces,
            targetPositions=[-pos, pos],
            positionGains=[0.1, 0.1],
        )
        self._target = width

    def _step(self, dt, vel):
        width = self._target
        self._move(width + vel * dt)

    def grasped(self):
        return False

    def opened(self):
        return self._gripper.width > 0.99


class RG6GripperController(object):
    def __init__(self, gripper):
        self._gripper = gripper
        self._velocity = 2
        self._force = 10.0
        self._gain = 1.0
        self._blocked_moment = 0.1
        self._kinematic_grasp = True
        self._kinematic_constraint = None

        self._left_block = False
        self._right_block = False
        self._object_grasped = False

    def reset(self, close=False):
        self._detach_grasped_body()
        self._kinematic_constraint = None
        self._target_velocity = 0
        position = self._gripper.joints.state.positions
        if close:
            self._state = GraspState.Closed
        else:
            self._state = GraspState.Opened
        self._move_to(position)
        self._left_block = False
        self._right_block = False
        self._history_position = []
        self._object_grasped = False

    def grasped(self):
        return self._object_grasped

    def opened(self):
        return self._state == GraspState.Opened

    @property
    def kinematic_grasp(self):
        return self._kinematic_grasp

    @kinematic_grasp.setter
    def kinematic_grasp(self, value):
        """If a kinematic grasp flag is set an object fixing
        in a gripper after closing."""
        self._kinematic_grasp = value

    @property
    def target_velocity(self):
        return self._target_velocity

    @target_velocity.setter
    def target_velocity(self, value):
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

    def step(self, dt):
        position = np.array(self._target_position.copy())
        if GraspState.Opening == self._state:
            if self._kinematic_grasp:
                self._detach_grasped_body()
            if self._history_position:
                position = self._history_position.pop()
            else:
                self._state = GraspState.Opened

        elif GraspState.Closing == self._state:
            self._history_position.append(position.copy())

            step = self._compute_step(dt, self._target_velocity)

            if any(step):

                position = np.clip(
                    position - step,
                    self._gripper._min_limit,
                    self._gripper._max_limit,
                )
                if self._kinematic_grasp:
                    self._attach_grasped_body()
                    if self._object_grasped:
                        self._state = GraspState.Closed
            else:
                self._state = GraspState.Closed

        if np.any(position != self._target_position):
            self._move_to(position)

    def _compute_step(self, dt, vel):
        return (
            vel
            * dt
            * np.array(
                [int(not self._left_block)] * 4 + [int(not self._right_block)] * 4
            )
        )

    def _step(self, dt, vel):
        position = np.array(self._target_position.copy())
        next_position = np.clip(
            position
            + (
                vel
                * dt
                * np.array(
                    [int(not self._left_block)] * 4 + [int(not self._right_block)] * 4
                )
            ),
            self._gripper._min_limit,
            self._gripper._max_limit,
        )
        self._move_to(next_position)

    def _move_to(self, position):
        self._gripper.joints.control(
            controlMode=pb.POSITION_CONTROL,
            targetPositions=position,
            positionGains=[self._gain] * len(self._gripper.joints),
            forces=[self._force] * len(self._gripper.joints),
        )
        self._target_position = position

    def _attach_grasped_body(self):
        grasping_frame = self._gripper._grasping_frame
        contacts = self._gripper.get_contacts()

        left_p0, left_o0 = self._gripper._left_tip.state.world_link_frame_position

        left_finger_contacts = [
            c
            for c in contacts
            if c.link_a == self._gripper._left_tip and c.body_b != self._gripper._body
        ]

        left_contacts = set()

        if left_finger_contacts:
            left_f = [
                np.multiply(c.contact_normal_on_b, c.normal_force)
                for c in left_finger_contacts
            ]
            left_p = [c.position_on_a for c in left_finger_contacts]
            left_moment = np.sum(np.abs(np.cross(left_f, np.subtract(left_p, left_p0))))
            if left_moment > self._blocked_moment:
                links = set([c.link_b for c in left_finger_contacts])
                left_contacts = links
                self._left_block = True
        else:
            left_contacts = set()
            self._left_block = False

        (
            right_p0,
            right_o0,
        ) = self._gripper._right_tip.state.world_link_frame_position

        right_finger_contacts = [
            c
            for c in contacts
            if c.link_a == self._gripper._right_tip and c.body_b != self._gripper._body
        ]

        right_contacts = set()

        if right_finger_contacts:
            right_f = [
                np.multiply(c.contact_normal_on_b, c.normal_force)
                for c in right_finger_contacts
            ]
            right_p = [c.position_on_a for c in right_finger_contacts]
            right_moment = np.sum(
                np.abs(np.cross(right_f, np.subtract(right_p, right_p0)))
            )
            if right_moment > self._blocked_moment:
                links = set([c.link_b for c in right_finger_contacts])
                right_contacts = links
                self._right_block = True

        else:
            right_contacts = set()
            self._right_block = False

        contacted_links = left_contacts.union(right_contacts)

        if contacted_links:
            link = contacted_links.pop()
            self._kinematic_constraint = Constraint.create_fixed(
                self._gripper._grasping_frame, link
            )
            for jt in self._gripper.joints:
                jt.child_link.dynamics.contact_constraint = (0.1, 0.1)
            self._object_grasped = True

    def _detach_grasped_body(self):
        if self._kinematic_constraint is not None:
            self._kinematic_constraint.remove()
            self._kinematic_constraint = None

            self._left_block = False
            self._right_block = False
            self._object_grasped = False

            for jt in self._gripper.joints:
                jt.child_link.dynamics.contact_constraint = (100, 100)
