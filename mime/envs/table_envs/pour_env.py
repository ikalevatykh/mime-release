from .pour_scene import PourScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv


class PourEnv(TableEnv):
    """ Pouring environment, trajectory observation, linear tool control """

    def __init__(self, **kwargs):
        scene = PourScene(**kwargs)
        super(PourEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            'tool_orientation', 'linear_velocity', 'angular_velocity',
            'grip_forces', 'grip_width')
        self.action_space = self._make_dict_space(
            'linear_velocity',
            'angular_velocity',
            # 'joint_velocity',
            'grip_velocity')

    def _get_observation(self, scene):
        tool_state = scene.robot.arm.tool.state
        return dict(
            tool_orientation=tool_state.position[1],
            linear_velocity=tool_state.velocity[0],
            angular_velocity=tool_state.velocity[1],
            grip_state=scene.robot.gripper.controller.state)


class PourCamEnv(TableCamEnv):
    """ Pick environment, camera observation, linear tool control """

    def __init__(self, view_rand, gui_resolution, cam_resolution, num_cameras,
                 **kwargs):
        scene = PourScene(**kwargs)
        super(PourCamEnv, self).__init__(
            scene, view_rand, gui_resolution, cam_resolution, num_cameras)

        self.action_space = self._make_dict_space(
            'linear_velocity',
            'angular_velocity',
            # 'joint_velocity',
            'grip_velocity')
