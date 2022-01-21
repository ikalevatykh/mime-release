from pyquaternion import Quaternion

import numpy as np
import pybullet as pb


class Script(object):
    def __init__(self, scene):
        self._dt = scene.dt
        self._max_tool_velocity = scene.max_tool_velocity
        self._max_gripper_velocity = scene.max_gripper_velocity

        # report used scripts to the scene
        self.scene = scene

    def joint_move(self, arm, pos=None, orn=None, t_acc=1.0, script_id=None):
        """Move in joint space with trap velocity profile"""
        pos0, orn0 = arm.controller.tool_target
        if pos is None:
            pos = pos0
        if orn is None:
            orn = orn0
        elif len(orn) == 3:
            orn = pb.getQuaternionFromEuler(orn)

        q0 = np.array(arm.controller.joints_target)
        q = arm.kinematics.inverse(pos, orn, q0)
        assert q is not None, "Cannot find IK solution for target configuration"
        dq_max = np.array(arm.max_joint_velocity) * 0.15
        velocities = trap_velocity_profile([q - q0], [dq_max], self._dt, t_acc)
        for (dq,) in velocities:
            self._report_script_to_scene(script_id)
            q = np.array(arm.controller.joints_target)
            v, w = arm.kinematics.forward_vel(q, dq, self._dt)
            yield dict(linear_velocity=v, angular_velocity=w, joint_velocity=dq)

    def joint_step(
        self, arm, pos=None, orn=None, t_acc=1.0, tool_cs=True, script_id=None
    ):
        pos0, orn0 = arm.controller.tool_target
        if pos is not None:
            if tool_cs:
                pos = np.add(pos0, to_quat(orn0).rotate(pos))
            else:
                pos = np.add(pos0, pos)
        if orn is not None:
            if len(orn) == 3:
                orn = pb.getQuaternionFromEuler(orn)
            q0, q = to_quat(orn0), to_quat(orn)
            if tool_cs:
                orn = to_orn(q0 * q)
            else:
                orn = to_orn(q * q0)
        for a in self.joint_move(arm, pos, orn, t_acc):
            self._report_script_to_scene(script_id)
            yield a

    def tool_move(self, arm, pos=None, orn=None, t_acc=1.0, script_id=None):
        """Linear move with trap velocity profile"""
        max_v, max_w = self._max_tool_velocity
        pos0, orn0 = arm.controller.tool_target
        dist, axis, angle = np.zeros(3), np.zeros(3), 0.0
        if pos is not None:
            dist = np.subtract(pos, pos0)
        if orn is not None:
            if len(orn) == 3:
                orn = pb.getQuaternionFromEuler(orn)
            diff = to_quat(orn) * to_quat(orn0).inverse
            axis, angle = diff.get_axis(undefined=np.array([0, 0, 0])), diff.angle

        velocities = trap_velocity_profile(
            [dist, angle * axis], [max_v, max_w], self._dt, t_acc
        )

        for v, w in velocities:
            self._report_script_to_scene(script_id)
            q = np.array(arm.controller.joints_target)
            dq = arm.kinematics.inverse_vel(q, v, w, self._dt)
            yield dict(linear_velocity=v, angular_velocity=w, joint_velocity=dq)

    def tool_step(
        self, arm, pos=None, orn=None, t_acc=1.0, tool_cs=True, script_id=None
    ):
        pos0, orn0 = arm.controller.tool_target
        if pos is not None:
            if tool_cs:
                pos = np.add(pos0, to_quat(orn0).rotate(pos))
            else:
                pos = np.add(pos0, pos)
        if orn is not None:
            q0, q = to_quat(orn0), to_quat(orn)
            if tool_cs:
                orn = to_orn(q0 * q)
            else:
                orn = to_orn(q * q0)
        for a in self.tool_move(arm, pos, orn, t_acc):
            self._report_script_to_scene(script_id)
            yield a

    def grip_close(self, gripper, script_id=None):
        self._report_script_to_scene(script_id)
        yield dict(grip_velocity=-self._max_gripper_velocity)
        for _ in np.arange(0, 2, self._dt):
            if gripper.controller.grasped():
                break
            self._report_script_to_scene(script_id)
            yield dict(grip_velocity=-self._max_gripper_velocity)
        self._report_script_to_scene(script_id)
        yield dict(grip_velocity=0)

    def grip_open(self, gripper, script_id=None):
        for _ in np.arange(0, 5, self._dt):
            if gripper.controller.opened():
                break
            self._report_script_to_scene(script_id)
            yield dict(grip_velocity=self._max_gripper_velocity)

    def idle(self, num_steps, script_id=None):
        for _ in range(num_steps):
            self._report_script_to_scene(script_id)
            yield dict(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

    def color_change(self, cup, color):
        cup.color = color

    def _report_script_to_scene(self, script_id):
        if script_id is not None:
            self.scene.scripts_used.append(script_id)


def trap_velocity_profile(distances, max_velocities, dt, t_acc=1.0):
    t_max = [np.linalg.norm(d / v) for d, v in zip(distances, max_velocities)]

    t_dec = np.max(t_max)
    t_acc = np.min([t_acc, t_dec])
    t_end = t_dec + t_acc

    v_coeff = [d / t_dec for d in distances]

    vels = []
    for t in np.arange(0.0, t_end, dt) + dt:
        k = 1.0
        if t > t_end:
            k = 0.0
        elif t <= t_acc:
            k = t / t_acc
        elif t >= t_dec:
            k = 1 - (t - t_dec) / t_acc
        vels.append([k * v for v in v_coeff])
    return vels


def to_quat(orn):
    if len(orn) == 3:
        orn = pb.getQuaternionFromEuler(orn)
    return Quaternion(w=orn[3], x=orn[0], y=orn[1], z=orn[2])


def to_orn(q):
    return [q[1], q[2], q[3], q[0]]
