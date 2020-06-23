import numpy as np

from mime.envs.table_envs.table_scene import TableScene
from mime.envs.table_envs.table_modder import TableModder
from mime.envs.script import Script
from mime.envs.table_envs.utils import MeshLoader, sample_without_overlap
from mime.scene.body import Body


class BowlScene(TableScene):
    def __init__(self, **kwargs):
        super(BowlScene, self).__init__(**kwargs)
        self._target = None
        self.num_subtasks = 4
        self._modder = TableModder(self, self._randomize)

        v, w = self._max_tool_velocity
        self._max_tool_velocity = (2 * v, w)

        self.meshes = {
            'bowl': {
                # 'type': 'shapenet_02880940',
                'type': 'bowl/model_normalized.urdf',
                'size_range': [0.2, 0.26],
                'color': (1, 1, 1, 1),
                'mass': 1.0,
            },
            'cube': {
                'type': 'cube',
                'size_range': [0.04, 0.065],
                'color': (0, 1, 0, 1),
                'mass': 0.2
            },
        }

        # so the gripper does not penetrate the table
        self._workspace[0][2] = 0.03

        # by default the script function will produce the full action sequence to solve the task
        # however you can set `skill_data_collection=True` to execute a random skill in each episode
        # each skill will be reset with a sequence of previous skills applied before it
        # this sequence will have a random length between 0 and `self.subtask`
        self.skill_data_collection = False

        # temporary fix for the pybullet rendering/memory bug
        self._reset_counter = 0
        self._max_resets_before_reconnect = 300

        self.skill2name = {
            0: 'down_grasp_up',
            1: 'goto_bowl',
            2: 'goto_cube',
            3: 'goto_ball',
            -1: 'dummy_close',
        }

    def load(self):
        super(BowlScene, self).load()

    def reset(self, np_random):
        if self._reset_counter == self._max_resets_before_reconnect:
            super(BowlScene, self).close()
            self._connected = False
            for mesh in self.meshes.values():
                mesh['body'] = None
            self._reset_counter = 0
        super(BowlScene, self).reset(np_random)
        modder = self._modder

        # remove meshes present in the scene
        for mesh in self.meshes.values():
            if 'body' in mesh and mesh['body'] is not None:
                mesh['body'].remove()

        # move robot arm to a random position
        low, high = self.workspace
        low, high = np.array(low.copy()), np.array(high.copy())
        low += 0.05
        high -= 0.05
        tool_pos = np_random.uniform(low=low, high=high)
        self.robot.arm.reset_tool(tool_pos)

        # load and set cage to a random position
        modder.load_cage(np_random)

        meshes = self.meshes
        for name, mesh in meshes.items():
            size_range = {
                'low': mesh['size_range'][0],
                'high': mesh['size_range'][1]
            }
            if 'shapenet' in mesh['type']:
                loader = MeshLoader(self.client_id, [mesh['type'].replace('shapenet_', '')],
                                    egl=self._load_egl)
                mesh['size'] = np_random.uniform(**size_range)
                mesh['body'] = loader.get_mesh(np_random.randint(len(loader)),
                                               mesh['size'])
            else:
                mesh['body'], mesh['size'] = modder.load_mesh(
                    mesh['type'], size_range, np_random, mass=mesh['mass'])
            mesh['body'].color = mesh['color']

        # set meshes to random positions
        aabbs = []
        meshes_body = [('shapenet' in mesh['type'] or 'bowl' in mesh['type'], mesh['body']) for mesh in meshes.values()]
        np_random.shuffle(meshes_body)
        for is_shapenet, mesh in meshes_body:
            x_rot = np.pi/2 if is_shapenet else 0
            aabbs, _ = sample_without_overlap(mesh,
                                              aabbs,
                                              np_random,
                                              low=low,
                                              high=high,
                                              x_rot=x_rot,
                                              min_dist=0.03)

        # a history of the scripts used by the scripted agent
        self.scripts_used = []
        self._reset_counter += 1

        # set the skill sequence
        if not self.skill_data_collection:
            self.skill_sequence = [3, 0, 1, 2]
        else:
            skill_sequence = []
            # random closing at the beginning
            if np_random.randint(2):
                skill_sequence += [-1]
            # random goto
            skill_sequence += list(2 + np_random.randint(2, size=(2,)))
            # graps and goto
            skill_sequence += [0, 1]
            skill_sequence += list(2 + np_random.randint(1, size=(1,)))
            self.skill_sequence = skill_sequence
            # print(skill_sequence)

        # if self.skill_data_collection:
            # print('Skill sequence : {}'.format(
                # [self.skill2name[skill] for skill in self.skill_sequence]))

    def script(self):
        for i in self.skill_sequence:
            yield self.script_subtask(i)

    def script_subtask(self, subtask):
        arm = self.robot.arm
        grip = self.robot.gripper
        tool_pos = np.array(arm.tool.state.position[0])

        subtask_to_mesh = {2: 'bowl', 3: 'cube'}

        # 0 : down, grasp
        # 1 : go up
        # 2 : go to bowl and release
        # 3 : go to cube and release

        sc = Script(self)
        script = []
        # ! subtask ID of down_grasp_up should be kept to 0, cf skill_sequence definition
        if subtask <= 1:
            dist_arm_mesh = [np.linalg.norm(((mesh['body'].position[0] - tool_pos)[:2]))
                             for mesh in self.meshes.values()]
            dist_arm_mesh = [(name, np.linalg.norm(((mesh['body'].position[0] - tool_pos)[:2])))
                             for name, mesh in self.meshes.items()]
            mesh_min_dist = np.argmin([x[1] for x in dist_arm_mesh])
            mesh_pick_pos = np.array(self.meshes[dist_arm_mesh[mesh_min_dist][0]]['body'].position[0])+[0, 0, 0.01]
            if dist_arm_mesh[mesh_min_dist][1] < 0.01:
                if subtask == 0:
                    script = [
                        sc.tool_move(arm, mesh_pick_pos, script_id=subtask),
                        sc.grip_close(grip, script_id=subtask),
                    ]
                else:
                    script = [sc.tool_step(arm, [0, 0, -0.11], script_id=subtask)]
        elif subtask in range(2, 4):
            mesh_name = subtask_to_mesh[subtask]
            mesh = self.meshes[mesh_name]['body']
            mesh_pos = np.array(mesh.position[0])
            aabb_size = -np.subtract(*mesh.collision_shape.AABB)
            goto_pos = mesh_pos.copy()
            goto_pos[2] = aabb_size[2] +0.08
            if mesh_pos[2] < 0.1:
                script = [
                    sc.tool_move(arm,
                                 goto_pos,
                                 script_id=subtask),
                    sc.grip_open(grip, script_id=subtask)
                ]
        elif subtask == -1:
            script = [sc.grip_close(grip, script_id=subtask)]
        else:
            raise ValueError('Skill {} does not exist.'.format(subtask))
        if self.skill_data_collection:
            script += [sc.idle(3, script_id=subtask)]
        return script

    @property
    def cube_position(self):
        pos, _ = self.meshes['cube']['body'].position
        return pos

    @property
    def mesh_position(self):
        pos, _ = self.meshes['ball']['body'].position
        return pos

    @property
    def bowl_position(self):
        pos, _ = self.meshes['bowl']['body'].position
        return pos

    @property
    def distance_to_cube(self):
        tool_pos, _ = self.robot.arm.tool.state.position
        return np.subtract(self.cube_position, tool_pos)

    @property
    def distance_to_bowl(self):
        tool_pos, _ = self.robot.arm.tool.state.position
        return np.subtract(self.bowl_position, tool_pos)

    def get_reward(self, action):
        success_reward = 10 * float(self.is_task_success())
        return success_reward

    def is_task_success(self):
        # cube
        # top of the cube above the bowl middle position
        objs_higher_than_bowl = self.meshes['cube']['body'].collision_shape.AABB[1][2] > self.bowl_position[2] - 0.02
        dist_cube_to_bowl = np.linalg.norm(
            np.subtract(self.cube_position[:2], self.bowl_position[:2]))
        objs_inside_bowl = dist_cube_to_bowl < 0.04

        # objs_higher_than_bowl = True
        # objs_inside_bowl = True

        # ball
        # objs_higher_than_bowl = self.ball_position[2] > self.bowl_position[2] - 0.02 and objs_higher_than_bowl
        # dist_ball_to_bowl = np.linalg.norm(
        #     np.subtract(self.ball_position[:2], self.bowl_position[:2]))
        # objs_inside_bowl = dist_ball_to_bowl < 0.02 and objs_inside_bowl

        return objs_higher_than_bowl and objs_inside_bowl

    def is_task_failure(self):
        # first check the joints errors
        is_failure, error_message = super(BowlScene, self).is_task_failure()
        return is_failure, error_message

    def _step(self, dt):
        super(BowlScene, self)._step(dt)


def test_scene():
    from time import sleep
    scene = BowlScene()
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()
