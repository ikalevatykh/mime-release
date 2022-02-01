from .tower_scene import TowerScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv

import numpy as np

class TowerEnv(TableEnv):
    """ Tower environment, trajectory observation, linear tool control """

    def __init__(self, **kwargs):
        scene = TowerScene(**kwargs)
        super(TowerEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            'distance_to_target',
            'target_position',
            'linear_velocity',
            'grip_forces',
            'grip_width'
        )
        self.action_space = self._make_dict_space(
            'linear_velocity',
            # 'joint_velocity',
            'grip_velocity'
        )

    def _get_observation(self, scene):
        return dict(
            tool_position=scene.robot.arm.tool.state.position[0],
            cubes_position=scene.cubes_position,
            distance_to_cubes =scene.distance_to_cubes,
            linear_velocity=scene.robot.arm.tool.state.velocity[0],
            grip_state=scene.robot.gripper.controller.state,)


class TowerCamEnv(TableCamEnv):
    """ Tower environment, camera observation, linear tool control """

    def __init__(self, view_rand, gui_resolution, cam_resolution, num_cameras, **kwargs):
        scene = TowerScene(**kwargs)
        super(TowerCamEnv, self).__init__(scene, view_rand, gui_resolution, cam_resolution, num_cameras)

        self.action_space = self._make_dict_space(
            'linear_velocity',
            # 'joint_velocity',
            'grip_velocity'
        )
