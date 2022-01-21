from .push_scene import PushScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv


class PushEnv(TableEnv):
    """Push environment, trajectory observation, linear tool control"""

    def __init__(self, **kwargs):
        scene = PushScene(**kwargs)
        super(PushEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            "distance_to_target",
            "target_position",
            "linear_velocity",
            "angular_velocity",
            "grip_forces",
            "grip_width",
        )
        self.action_space = self._make_dict_space(
            "linear_velocity",
            "angular_velocity",
            # 'joint_velocity',
            "grip_velocity",
        )

    def _get_observation(self, scene):
        obs_dic = super(PushEnv, self)._get_observation(scene)
        obs_dic.update(
            dict(
                distance_to_goal=scene.distance_to_target,
                target_position=scene.target_position,
                marker_position=scene.marker_position,
                target_orientation=scene.target_orientation,
            )
        )

        return obs_dic


class PushCamEnv(TableCamEnv):
    """Push environment, camera observation, linear tool control"""

    def __init__(
        self, view_rand, gui_resolution, cam_resolution, num_cameras, **kwargs
    ):
        scene = PushScene(**kwargs)
        super(PushCamEnv, self).__init__(
            scene, view_rand, gui_resolution, cam_resolution, num_cameras
        )

        self.action_space = self._make_dict_space(
            "linear_velocity",
            "angular_velocity",
            # 'joint_velocity',
            "grip_velocity",
        )

    def _get_observation(self, scene):
        obs_dic = super(PushCamEnv, self)._get_observation(scene)
        obs_dic.update(
            dict(
                distance_to_goal=scene.distance_to_target,
                target_position=scene.target_position,
                marker_position=scene.marker_position,
                target_orientation=scene.target_orientation,
            )
        )

        return obs_dic
