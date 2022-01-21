import numpy as np
import pybullet as pb  # only used for euler2quat

from math import pi
from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script
from mime.config import assets_path
from .utils import load_textures
from mime.scene.body import Body

OBJ_TEXTURES_PATH = assets_path() / "textures" / "objects" / "simple"


class RopeScene(TableScene):
    def __init__(self, **kwargs):
        super(RopeScene, self).__init__(**kwargs)
        self._target = None

        # linear velocity x2 for the real setup
        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)
        self.rope_radius = 0.01
        self.n_parts = 14
        self.length_rope = 2 * self.rope_radius * self.n_parts * np.sqrt(2)
        self._rope = []

    def load(self, np_random):
        super(RopeScene, self).load(np_random)

    def load_textures(self, np_random):
        super().load_textures(np_random)
        self._modder._textures["objects"] = load_textures(OBJ_TEXTURES_PATH, np_random)

    def reset(
        self,
        np_random,
    ):
        """
        Reset the cube position and arm position.
        """

        super(RopeScene, self).reset(np_random)
        modder = self._modder
        modder.load_cage(np_random)

        if self._domain_rand:
            modder.randomize_cage_visual(np_random)

        # define workspace, tool position and cube position
        low, high = self.workspace
        low, high = np.array(low.copy()), np.array(high.copy())

        if self._target is not None:
            self._target.remove()

        x_gripper_min, x_gripper_max = (
            self._workspace[0][0],
            self._workspace[1][0],
        )
        y_gripper_min, y_gripper_max = (
            self._workspace[0][1],
            self._workspace[1][1],
        )
        gripper_pos = [
            np_random.uniform(x_gripper_min, x_gripper_max),
            np_random.uniform(y_gripper_min, y_gripper_max),
            np_random.uniform(self._safe_height[0], self._safe_height[1]),
        ]

        if self._robot_type == "PRL_UR5":
            gripper_orn = [pi, 0, pi / 2]
        else:
            gripper_orn = None

        q0 = self.robot.arm.controller.joints_target
        q = self.robot.arm.kinematics.inverse(gripper_pos, gripper_orn, q0)
        self.robot.arm.reset(q)

        # load rope, set to random size and random position
        rope_pos = np_random.uniform(low=low, high=high)
        rope_pos[2] = 0.02
        self._rope = Body.rope(
            rope_pos,
            self.length_rope,
            self.client_id,
            n_parts=self.n_parts,
            color=[0, 0.7, 0.5],
            color_ends=[0, 0.7, 0.5],
            radius=self.rope_radius,
            egl=self._load_egl,
        )
        self._target = self._rope.parts[0]
        for i in range(1000):
            pb.stepSimulation(physicsClientId=self.client_id)

        self._marker = Body.mesh(
            "plane.obj", self.client_id, scale=[0.05, 0.05, 1], egl=self._load_egl
        )

        self._marker.position = [(0, 0, 0.0001), (0, 0, 0, 1)]
        self._marker.color = [1, 0, 0, 1]

    def compute_goal(self):
        middle_part = self._rope.parts[int(len(self._rope.parts) / 2)]
        middle_position, _ = middle_part.position
        distance = self._rope.distance
        length = self._rope.length
        parts_goal_position = []
        for i in range(len(self._rope.parts)):
            part_goal_position = np.array(middle_position).copy()
            part_goal_position[0] += distance * i - length / 2
            parts_goal_position.append(part_goal_position)
        return parts_goal_position

    def script(self):
        """
        Script to generate expert demonstrations.
        """
        arm = self.robot.arm
        grip = self.robot.gripper
        parts_goal_position = self.compute_goal()
        z_offset = self._rope.radius + 0.01
        pick_pos_end1, _ = self._rope.parts[0].position
        pick_pos_end1 = np.array(pick_pos_end1)
        pick_pos_end2, _ = self._rope.parts[-1].position
        pick_pos_end2 = np.array(pick_pos_end2)
        sc = Script(self)
        return [
            sc.tool_move(arm, pick_pos_end1 + [0, 0, 0.1]),
            sc.tool_move(arm, pick_pos_end1 + [0, 0, z_offset]),
            sc.grip_close(grip),
            sc.tool_move(arm, pick_pos_end1 + [0, 0, 0.05]),
            sc.tool_move(arm, parts_goal_position[0] + [0, 0, 0.02]),
            sc.grip_open(grip),
            sc.tool_move(arm, pick_pos_end2 + [0, 0, 0.1]),
            sc.tool_move(arm, pick_pos_end2 + [0, 0, z_offset]),
            sc.grip_close(grip),
            sc.tool_move(arm, pick_pos_end2 + [0, 0, 0.05]),
            sc.tool_move(arm, parts_goal_position[-1] + [0, 0, 0.02]),
            sc.grip_open(grip),
        ]

    @property
    def target_position(self):
        pos_base, _ = self._robot._body.position
        pos_cube, _ = self._target.position
        return np.array(pos_cube) - np.array(pos_base)

    @property
    def target_orientation(self):
        _, orn_cube = self._target.position
        orn_cube_euler = pb.getEulerFromQuaternion(orn_cube)
        return np.array([orn_cube_euler[-1] / np.pi * 180])

    @property
    def distance_to_target(self):
        tool_pos, _ = self.robot.arm.tool.state.position
        return np.subtract(self.target_position, tool_pos)

    def get_reward(self, action):
        return 0

    def is_task_success(self):
        return False


def test_scene():
    from time import sleep

    scene = RopeScene(robot_type="PRL_UR5")
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        obs = scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()
