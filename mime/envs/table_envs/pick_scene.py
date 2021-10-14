import numpy as np
import pybullet as pb  # only used for euler2quat

from math import pi
from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script


class PickScene(TableScene):
    def __init__(self, **kwargs):
        super(PickScene, self).__init__(**kwargs)
        self._target = None
        self._modder = TableModder(self, self._randomize)

        # linear velocity x2 for the real setup
        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)

        self._cube_size_range = {"low": 0.05, "high": 0.05}

    def load(self):
        super(PickScene, self).load()

    def reset(
        self,
        np_random,
        gripper_pose=None,
        cube_pose=None,
    ):
        """
        Reset the cube position and arm position.
        """

        super(PickScene, self).reset(np_random)
        modder = self._modder

        # define workspace, tool position and cube position
        low, high = self.workspace
        low, high = np.array(low.copy()), np.array(high.copy())
        low[:2] += 0.05
        high[:2] -= 0.05

        if cube_pose is None:
            cube_pos = np_random.uniform(low=low, high=high)
        else:
            cube_pos, cube_ori = cube_pose

        if self._target is not None:
            self._target.remove()

        if gripper_pose is None:
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
        else:
            gripper_pos, gripper_orn = gripper_pose

        q0 = self.robot.arm.controller.joints_target
        q = self.robot.arm.kinematics.inverse(gripper_pos, gripper_orn, q0)
        self.robot.arm.reset(q)

        # load and set cage to a random position
        modder.load_cage(np_random)

        # load cube, set to random size and random position
        cube, cube_size = modder.load_mesh("cube", self._cube_size_range, np_random)

        cube.color = (11.0 / 151.0, 56.0 / 127.0, 60.0 / 255.0, 1)

        self._cube_size = cube_size
        self._target = cube
        self._target.position = (cube_pos[0], cube_pos[1], self._cube_size / 2)

    def script(self):
        """
        Script to generate expert demonstrations.
        """
        arm = self.robot.arm
        grip = self.robot.gripper
        pick_pos = np.array(self.target_position)

        sc = Script(self)
        return [
            sc.tool_move(arm, pick_pos + [0, 0, 0.1]),
            sc.tool_move(arm, pick_pos + [0, 0, 0.01]),
            sc.grip_close(grip),
            sc.tool_move(arm, pick_pos + [0, 0, 0.12]),
        ]

    @property
    def target_position(self):
        pos_base, _ = self._robot._body.position
        pos_cube, _ = self._target.position
        return np.array(pos_cube) - np.array(pos_base)

    @property
    def target_rotation(self):
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
        return self.target_position[2] > 0.08


def test_scene():
    from time import sleep

    scene = PickScene(robot_type="PRL_UR5")
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        obs = scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()
