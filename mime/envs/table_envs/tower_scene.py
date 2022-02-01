import numpy as np

from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script
from .utils import sample_without_overlap


class TowerScene(TableScene):
    def __init__(self, **kwargs):
        super(TowerScene, self).__init__(**kwargs)
        self._modder = TableModder(self, self._randomize)

        self._count_success = 0
        self._num_cubes = 2
        self._cubes = []
        self._cubes_size = []
        self._cube_size_range = {'low': 0.03, 'high': 0.085}

        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)

    def load(self):
        super(TowerScene, self).load()

    def reset(self, np_random):
        super(TowerScene, self).reset(np_random)
        modder = self._modder
        self._count_success = 0

        low, high = self._workspace
        low_arm, high_arm = low.copy(), high.copy()
        low_arm[2] = 0.1
        low_cubes, high_cubes = np.array(low.copy()), np.array(high.copy())
        low_cubes[:2] += 0.02
        high_cubes[:2] -= 0.02

        if self._cage is not None:
            self._cage.remove()
        for cube in self._cubes:
            cube.remove()
        self._cubes = []
        self._cubes_size = []

        init_qpos = self._lab_init_qpos
        self.robot.arm.reset(init_qpos)

        # load and set cage to a random position
        modder.load_cage(np_random)

        # load cubes
        num_cubes = self._num_cubes
        cubes = []
        cubes_size = []
        for i in range(num_cubes):
            cube, cube_size = modder.load_mesh(
                'cube', self._cube_size_range, np_random)
            # cube.dynamics.lateral_friction = 10
            cubes.append(cube)
            cubes_size.append(cube_size)
        # sort cubes per decreasing size
        # biggest cube first
        idxs_sort = np.argsort(-np.array(cubes_size))
        for idx in idxs_sort:
            self._cubes.append(cubes[idx])
            self._cubes_size.append(cubes_size[idx])
        self._cubes_size = np.array(self._cubes_size)

        # move cubes to a random position and change color
        cubes = []
        aabbs = []
        for cube in self._cubes:
            aabbs, _ = sample_without_overlap(cube, aabbs, np_random,
                    low_cubes, high_cubes, 0, 0, min_dist=0.05)
            color = np_random.uniform(0, 1, 4)
            color[3] = 1
            cube.color = color

    def script(self):
        arm = self.robot.arm
        grip = self.robot.gripper
        cubes_pos = self.cubes_position
        cubes_size = self._cubes_size
        tower_pos = cubes_pos[0]
        height = 0

        sc = Script(self)
        moves = []
        z_offset = 0.01
        for pick_pos, cube_size in zip(cubes_pos[1:], cubes_size[:-1]):
            height += cube_size
            moves += [
                sc.tool_move(arm, pick_pos + [0, 0, 0.1]),
                sc.tool_move(arm, pick_pos + [0, 0, z_offset]),
                sc.grip_close(grip),
                sc.tool_move(arm, pick_pos + [0, 0, height+z_offset+0.02]),
                sc.tool_move(arm, tower_pos + [0, 0, height+z_offset+0.02]),
                sc.grip_open(grip),
                sc.tool_move(arm, tower_pos + [0, 0, height+cube_size]),]

        return moves

    @property
    def cubes_position(self):
        return np.array([cube.position[0] for cube in self._cubes])

    def distance_to_cubes(self, idx):
        tool_pos, _ = self.robot.arm.tool.state.position
        return np.array([np.subtract(cube.position[0], tool_pos) for cube in self._cubes])

    def get_reward(self, action):
        return 0

    def is_task_success(self):
        cubes_pos = self.cubes_position
        cubes_size = self._cubes_size
        tower_pos = cubes_pos[0]
        heights = np.cumsum(cubes_size)[1:]
        bary_cubes = np.mean(cubes_pos[1:], axis=0)
        cubes_on_tower = np.linalg.norm(np.subtract(tower_pos[:2], bary_cubes[:2])) < 0.04

        heights_ok = True
        for cube_pos, cube_size, height in zip(cubes_pos[1:], cubes_size[1:], heights):
            heights_ok = heights_ok and np.abs(cube_pos[2]+cube_size/2-height) < 0.001

        if heights_ok and cubes_on_tower:
            self._count_success += 1
        success = self._count_success > 5
        return success

def test_scene():
    from time import sleep
    scene = TowerScene()
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()
