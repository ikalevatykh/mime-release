from .bowl_scene import BowlScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv

class BowlEnv(TableEnv):
    """ Bowl environment, trajectory observation, linear tool control """

    def __init__(self, **kwargs):
        scene = BowlScene(**kwargs)
        super(BowlEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            'tool_position',
            'linear_velocity',
            'grip_velocity',
            'cube_position',
            'bowl_position',
            'distance_to_cube',
            'distance_to_bowl',
            'skill'
        )
        self.action_space = self._make_dict_space(
            'linear_velocity',
            'grip_velocity'
        )

    @staticmethod
    def get_full_state_dict(scene):
        tool_state = scene.robot.arm.tool.state
        if len(scene.scripts_used):
            skill_used = scene.scripts_used[-1]
        else:
            skill_used = -1
        return dict(
            tool_position=scene.robot.arm.tool.state.position[0],
            linear_velocity=tool_state.velocity[0],
            grip_velocity=scene.robot.gripper.controller.state,
            cube_position=scene.cube_position,
            bowl_position=scene.bowl_position,
            distance_to_cube=scene.distance_to_cube,
            distance_to_bowl=scene.distance_to_bowl,
            skill=skill_used)

    def _get_observation(self, scene):
        return BowlEnv.get_full_state_dict(scene)


class BowlCamEnv(TableCamEnv):
    """ Pouring environment, camera observation, linear tool control """

    def __init__(self, view_rand, gui_resolution, cam_resolution, num_cameras, **kwargs):
        scene = BowlScene(**kwargs)
        super(BowlCamEnv, self).__init__(scene, view_rand, gui_resolution, cam_resolution, num_cameras)

        self.action_space = self._make_dict_space(
            'linear_velocity',
            'grip_velocity'
        )

    def _get_observation(self, scene):
        observ = super(BowlCamEnv, self)._get_observation(scene)
        full_state_observ = BowlEnv.get_full_state_dict(scene)
        observ.update(full_state_observ)
        # remove some observtaions to be consistent with BowlEnv
        observ.pop('joint_position')
        observ.pop('angular_velocity')
        observ.pop('tool_orientation')
        return observ
