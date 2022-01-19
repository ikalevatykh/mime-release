import numpy as np
import pybullet as pb  # only used for euler2quat

from math import pi
from enum import Enum
from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script
from mime.config import assets_path
from .utils import load_textures

OBJ_TEXTURES_PATH = assets_path() / "textures" / "objects" / "simple"


class Target(Enum):
    CUBE = 1
    CYLINDER = 2


class PushScene(TableScene):
    def __init__(self, **kwargs):
        super(PushScene, self).__init__(**kwargs)
        self._target = None

        # linear velocity x2 for the real setup
        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)

        self._target_type = Target.CYLINDER

        if self._target_type == Target.CYLINDER:
            if self._rand_obj:
                radius_range = {"low": 0.03, "high": 0.07}
                height_range = {"low": 0.06, "high": 0.08}
                self._cylinder_size_range = {
                    "low": [radius_range["low"], height_range["low"]],
                    "high": [radius_range["high"], height_range["high"]],
                }
            else:
                height = 0.085
                radius = 0.0425
                self._cylinder_size_range = [radius, height]
        elif self._target_type == Target.CUBE:
            if self.rand_obj:
                self._cube_size_range = {"low": 0.03, "high": 0.06}
            else:
                self._cube_size_range = 0.05
        else:
            raise ValueError(f"Target Type {self._target_type} is not valid.")

        self._marker_size_range = 0.07

    def load(self, np_random):
        super(PushScene, self).load(np_random)

    def load_textures(self, np_random):
        super().load_textures(np_random)
        self._modder._textures["objects"] = load_textures(OBJ_TEXTURES_PATH, np_random)

    def reset(
        self,
        np_random,
    ):
        """
        Reset the target and arm position.
        """

        super(PushScene, self).reset(np_random)
        modder = self._modder

        # load and randomize cage
        modder.load_cage(np_random)
        if self._domain_rand:
            modder.randomize_cage_visual(np_random)

        # define workspace, tool position and cylinder position
        low, high = self._object_workspace
        low, high = np.array(low.copy()), np.array(high.copy())

        if self._target is not None:
            self._target.remove()

        gripper_pos, gripper_orn = self.random_gripper_pose(np_random)

        q0 = self.robot.arm.controller.joints_target
        q = self.robot.arm.kinematics.inverse(gripper_pos, gripper_orn, q0)
        self.robot.arm.reset(q)

        # load cube, set to random size and random position
        if self._target_type == Target.CUBE:
            target, target_size = modder.load_mesh(
                "cube", self._cube_size_range, np_random
            )
            target_height = target_size
            target_width = target_size * 2
        elif self._target_type == Target.CYLINDER:
            target, target_size = modder.load_mesh(
                "cylinder", self._cylinder_size_range, np_random
            )
            target_height = target_size[1]
            target_width = target_size[0] * 2
        else:
            raise ValueError(f"Target Type {self._target_type} is not valid.")

        target_pos = np_random.uniform(low=low, high=high)

        target_color = (54.0 / 255.0, 73.0 / 255.0, 150.0 / 255.0, 1.0)
        target.color = target_color
        self._target_height = target_height
        self._target_width = target_width
        self._target = target
        self._target.position = (target_pos[0], target_pos[1], self._target_height / 2)

        # Create Marker
        marker, marker_size = modder.add_marker(
            "square", self._marker_size_range, np_random
        )
        marker_pos = np_random.uniform(
            low=low + marker_size / 2, high=high - marker_size / 2
        )

        while np.linalg.norm(marker_pos[:2] - target_pos[:2]) < self._target_width * 3:
            marker_pos = np_random.uniform(
                low=low + marker_size / 2, high=high - marker_size / 2
            )

        marker_pos[2] = 0.0001
        marker_color = [178 / 255, 0, 0, 1]
        self._marker = marker
        self._marker.position = marker_pos
        self._marker_position = marker_pos
        self._marker.color = marker_color
        if self._domain_rand:
            modder.randomize_object_color(np_random, target, target_color)
            modder.randomize_object_color(np_random, marker, marker_color)

    def script(self):
        """
        Script to generate expert demonstrations.
        """
        arm = self.robot.arm
        grip = self.robot.gripper
        target_pos = np.array(self.target_position)
        marker_pos = np.array(self._marker_position)

        def_gripper_orn = [pi, 0, pi / 2]

        push_v = marker_pos[:2] - target_pos[:2]
        push_distance = np.linalg.norm(push_v)
        push_norm_v = push_v / push_distance
        ref_v = np.array([1, 0]) if push_norm_v[0] > 0 else np.array([-1, 0])
        push_angle = np.arccos(
            np.dot(ref_v, push_norm_v)
            / (np.linalg.norm(ref_v) * np.linalg.norm(push_norm_v))
        )

        push_angle = push_angle if push_norm_v[0] * push_norm_v[1] > 0 else -push_angle

        init_push_xy = target_pos[:2] - push_norm_v * self._target_width / 1.2
        init_push_pos = np.array(
            [init_push_xy[0], init_push_xy[1], self._target_height]
        )

        init_push_orn = np.array(
            [
                def_gripper_orn[0],
                def_gripper_orn[1],
                def_gripper_orn[2] + push_angle,
            ]
        )

        sc = Script(self)
        end_push_xy = marker_pos[:2] - push_norm_v * (self._target_width / 2)
        end_push_pos = np.array([end_push_xy[0], end_push_xy[1], self._target_height])

        return [
            sc.grip_close(grip),
            sc.tool_move(arm, init_push_pos + [0, 0, 0.08], orn=init_push_orn),
            sc.tool_move(arm, init_push_pos),
            sc.tool_move(arm, end_push_pos),
        ]

    @property
    def target_position(self):
        pos_target, _ = self._target.position
        return np.array(pos_target)

    @property
    def marker_position(self):
        pos_marker, _ = self._marker.position
        return np.array(pos_marker)

    @property
    def target_orientation(self):
        _, orn_cube = self._target.position
        orn_cube_euler = pb.getEulerFromQuaternion(orn_cube)
        return np.array([orn_cube_euler[-1] / np.pi * 180])

    @property
    def distance_to_target(self):
        return np.subtract(self.marker_position, self.target_position)

    def get_reward(self, action):
        return 0

    def is_task_success(self):
        return np.linalg.norm(self.distance_to_target[:2]) < 0.015


def test_scene():
    from time import sleep

    scene = PushScene(robot_type="PRL_UR5")
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        obs = scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()
