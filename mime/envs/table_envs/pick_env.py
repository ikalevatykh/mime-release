from .pick_scene import PickScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv


class PickEnv(TableEnv):
    """ Pick environment, trajectory observation, linear tool control """

    def __init__(self, **kwargs):
        scene = PickScene(**kwargs)
        super(PickEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            'distance_to_target', 'target_position', 'linear_velocity',
            'grip_forces', 'grip_width')
        self.action_space = self._make_dict_space(
            'linear_velocity',
            # 'joint_velocity',
            'grip_velocity',
        )

    def _get_observation(self, scene):
        obs_dic = super(PickEnv, self)._get_observation(scene)
        obs_dic.update(
            dict(
                distance_to_goal=scene.distance_to_target,
                target_position=scene.target_position,
            ))

        return obs_dic


class PickCamEnv(TableCamEnv):
    """ Pick environment, camera observation, linear tool control """

    def __init__(self, view_rand, gui_resolution, cam_resolution, num_cameras,
                 **kwargs):
        scene = PickScene(**kwargs)
        super(PickCamEnv, self).__init__(scene, view_rand, gui_resolution,
                                         cam_resolution, num_cameras)

        self.action_space = self._make_dict_space(
            'linear_velocity',
            # 'joint_velocity',
            'grip_velocity')

    def _get_observation(self, scene):
        obs_dic = super(PickCamEnv, self)._get_observation(scene)
        obs_dic.update(
            dict(
                distance_to_goal=scene.distance_to_target,
                target_position=scene.target_position,
            ))

        return obs_dic
