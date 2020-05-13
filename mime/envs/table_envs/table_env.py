from ..scene_env import SceneEnv
from .. import utils


class TableEnv(SceneEnv):
    """ Base environment for table scene tasks """

    def __init__(self, scene):
        super(TableEnv, self).__init__(scene)

    def _get_observation(self, scene):
        obs_dic = dict(
            joint_position=scene.robot.arm.joints_position,
            tool_position=scene.robot.arm.tool.state.position[0],
            tool_orientation=scene.robot.arm.tool.state.position[1],
            linear_velocity=scene.robot.arm.tool.state.velocity[0],
            angular_velocity=scene.robot.arm.tool.state.velocity[1],
            grip_velocity=scene.robot.gripper.controller.state)

        return obs_dic

    def _set_action(self, scene, action):
        if 'linear_velocity' in action or 'angular_velocity' in action:
            v = action.get('linear_velocity', None)
            w = action.get('angular_velocity', None)
            scene.robot.arm.controller.tool_target_velocity = v, w
        if 'joint_velocity' in action:
            dq = action['joint_velocity']
            scene.robot.arm.controller.joints_target_velocity = dq
        if 'grip_velocity' in action:
            g = action['grip_velocity']
            scene.robot.gripper.controller.target_velocity = g
        if 'tool_position' in action or 'tool_orientation' in action:
            p = action.get('tool_position', None)
            o = action.get('tool_orientation', None)
            scene.robot.arm.controller.tool_target = p, o

    def _make_dict_space(self, *keys):
        return utils.make_dict_space(self._scene, *keys)
