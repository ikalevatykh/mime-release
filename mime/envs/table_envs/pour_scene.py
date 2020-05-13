import numpy as np
import pybullet as pb

from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script
from .utils import MeshLoader, aabb_collision, sample_without_overlap


class PourScene(TableScene):
    def __init__(self, **kwargs):
        super(PourScene, self).__init__(**kwargs)
        self._modder = TableModder(self, self._randomize)

        self._train = True
        self._workspace = [[0.44, -0.35, 0.1], [0.7, 0.35, 0.30]]
        self._bottle = None
        self._bowl = None
        self._drops = []
        self._drops_count = 5
        self._done_counter = 0
        self._scene_objs = []

        self._objs_size_range = {'low': 0.2, 'high': 0.26}

        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, 2 * w)

    def load(self):
        super(PourScene, self).load()

        # mesh loaders
        self._bottle_loader = MeshLoader(self.client_id, ['02876657'], [1], self._train,
                                         simplified=True, egl=self._load_egl)
        self._bowl_loader = MeshLoader(self.client_id, ['02880940'], [1], self._train,
                                       simplified=False, egl=self._load_egl)
        # load drops once
        self._drops = []
        for _ in range(self._drops_count):
            drop = self._modder.load_mesh('sphere', 0.02, None, mass=1e-6)[0]
            drop.color = (0.0, 1.0, 0.0, 1.0)
            self._drops.append(drop)

    def reset(self, np_random):
        super(PourScene, self).reset(np_random)
        bowl_loader = self._bowl_loader
        bottle_loader = self._bottle_loader
        self._done_counter = 0

        if self._scene_objs:
            for obj in self._scene_objs:
                obj.remove()
            self._scene_objs = []
        if self._cage is not None:
            self._cage.remove()

        # move arm to a predefined position
        # self.robot.arm.reset_tool((0.51, -0.0, 0.29), (np.pi / 2, 0, np.pi / 2))
        self.robot.arm.reset_tool((0.52, 0.0, 0.3),
                                  (np.pi / 2, 0, np.pi / 2))
        self.robot.gripper.reset('Basic')
        low_workspace, high_workspace = self._workspace

        # move drops away
        self._drop_counter = 0
        self._step_counter = 0
        for drop in self._drops:
            drop.position = [1e6, 1e6, 1e6]

        # load and set cage to a random position
        self._modder.load_cage(np_random)

        # load bottle
        objs_low_range = self._objs_size_range['low']
        objs_high_range = self._objs_size_range['high']
        bottle_size = np_random.uniform(objs_low_range, objs_high_range)
        bottle = bottle_loader.get_mesh(np_random.randint(len(bottle_loader)),
                                        bottle_size)
        bottle.color = (0, 1, 1, 1)
        bottle.dynamics.lateral_friction = 10
        self._bottle = bottle
        self._bottle_size = bottle_size
        self._bottle_dims = np.ptp(
            np.array(bottle.collision_shape.AABB), axis=0)

        # load bowl
        bowl_size = np_random.uniform(objs_low_range, objs_high_range)
        bowl = bowl_loader.get_mesh(np_random.randint(len(bowl_loader)),
                                    bowl_size)
        bowl.color = (1, 0, 1, 1)
        self._bowl = bowl

        self._scene_objs = [bottle, bowl]

        # set objects in the scenes
        aabbs = []
        for obj in self._scene_objs:
            aabbs, _ = sample_without_overlap(
                obj, aabbs, np_random,
                low=np.subtract(low_workspace, [-0.1, -0.1, 0]),
                high=np.subtract(high_workspace, [0.05, 0.1, 0]),
                low_z_rot=0,
                high_z_rot=2*np.pi,
                x_rot=np.pi/2,
                min_dist=0.02)

        self._drop_pos = np.array(bowl.position[0])

    def script(self):
        arm = self.robot.arm
        grip = self.robot.gripper
        bottle_pos = np.array(self.bottle_position)
        bowl_pos = np.array(self.bowl_position)

        sc = Script(self)
        return [
            sc.tool_move(arm, pos=bottle_pos + [0.005, 0.0, 0.17]),
            sc.tool_move(arm, pos=bottle_pos + [0.005, 0.0, 0.0]),
            sc.tool_step(arm, pos=[0.0, 0.0, 0.05]),
            sc.grip_close(grip),
            sc.tool_move(arm, pos=bottle_pos + [0, 0, 0.14]),
            sc.tool_move(arm, pos=bowl_pos +
                         [0.05, -self._bottle_dims[1] / 2 - 0.02, 0.2]),
            sc.tool_step(arm, orn=[0, 0, -np.pi / 1.5]),
            sc.idle(num_steps=10)
        ]

    @property
    def bottle_position(self):
        pos, _ = self._bottle.position
        return pos

    @property
    def bowl_position(self):
        pos, _ = self._bowl.position
        return pos

    @property
    def drop_position(self):
        pos, _ = self._drop.position
        return pos

    def update_drops(self):
        rot, _, _ = pb.getEulerFromQuaternion(self._bottle.position[1])
        if rot < 0:
            drop = self._drops[self._drop_counter]
            self._drop_counter += 1
            z = 0.2 + (self._bottle_size / 2) * np.sin(rot)
            drop.position = self._drop_pos + [0, 0, z]

    def _step(self, dt):
        super(PourScene, self)._step(dt)
        self._step_counter += 1
        if (self._step_counter % 100) == 0:
            if self._drop_counter < len(self._drops):
                self.update_drops()

    def get_reward(self, action):
        return 0

    def is_task_success(self):
        bowl_aabb = self._bowl.collision_shape.AABB
        done = all([
            aabb_collision(bowl_aabb, d.collision_shape.AABB)
            and d.position[0][2] < 0.12 for d in self._drops
        ])
        if done:
            self._done_counter += 1

        return self._done_counter > 1

    def is_task_failure(self):
        failure, msg = super(PourScene, self).is_task_failure()
        if failure:
            return failure, msg
        rot, _, _ = pb.getEulerFromQuaternion(self._bottle.position[1])
        fallen_bottle = min(abs(np.mod(rot, np.pi)),
                            abs(np.mod(np.pi-rot, np.pi))) < np.pi/10 and self._bottle.position[0][2] < 0.05
        return fallen_bottle, 'Fallen bottle'


def test_scene():
    from itertools import cycle
    from time import sleep
    import numpy as np

    scene = PourScene(robot_type='UR5')
    scene.renders(True)
    scene.reset(np.random)
    workspace = np.array(scene.workspace)

    scene.step()

    pts = cycle([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1],
                 [1, 1, 1], [0, 1, 1], [0, 0, 1]])
    for ind in pts:
        pos = workspace[ind[0], 0], workspace[ind[1], 1], workspace[ind[2], 2]
        sc = Script(scene)
        script = sc.tool_move(
            scene.robot.arm, pos=pos, orn=(np.pi / 2, 0, np.pi / 2))

        act = next(script, None)
        while act is not None:
            v = act.get('linear_velocity', None)
            w = act.get('angular_velocity', None)
            scene.robot.arm.controller._tool_step(scene.dt, v, w)
            scene.step()
            act = next(script, None)

        sleep(1)


if __name__ == "__main__":
    test_scene()
