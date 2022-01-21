from .sweep_scene import SweepScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv


class RopeEnv(SweepEnv):
    """Pick environment, trajectory observation, linear tool control"""

    def __init__(self, **kwargs):
        scene = SweepScene(**kwargs)
        super(SweepEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            "distance_to_target",
            "target_position",
            "linear_velocity",
            "grip_forces",
            "grip_width",
        )
        self.action_space = self._make_dict_space(
            "linear_velocity",
            "grip_velocity",
        )

    def _get_observation(self, scene):
        obs_dic = super(SweepEnv, self)._get_observation(scene)
        obs_dic.update(
            dict(
                distance_to_goal=scene.distance_to_target,
                target_position=scene.target_position,
                target_orientation=scene.target_orientation,
            )
        )

        return obs_dic


class SweepCamEnv(TableCamEnv):
    """Pick environment, camera observation, linear tool control"""

    def __init__(
        self, view_rand, gui_resolution, cam_resolution, num_cameras, **kwargs
    ):
        scene = SweepScene(**kwargs)
        super(SweepCamEnv, self).__init__(
            scene, view_rand, gui_resolution, cam_resolution, num_cameras
        )

        self.action_space = self._make_dict_space(
            "linear_velocity",
            "grip_velocity",
        )

    def _get_observation(self, scene):
        obs_dic = super(SweepCamEnv, self)._get_observation(scene)
        obs_dic.update(
            dict(
                distance_to_goal=scene.distance_to_target,
                target_position=scene.target_position,
                target_orientation=scene.target_orientation,
            )
        )

        return obs_dic
