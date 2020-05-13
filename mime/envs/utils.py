import numpy as np
from gym import spaces


def make_dict_space(scene, *keys):
    """ Generate gym dict space for standard signals """

    items = {}
    for key in keys:
        discrete = False
        dtype = np.float32
        if 'linear_velocity' == key:
            hi = np.full(3, scene.max_tool_velocity[0])
            lo = -hi
        elif 'angular_velocity' == key:
            hi = np.full(3, scene.max_tool_velocity[1])
            lo = -hi
        elif 'tool_orientation' == key:
            hi = np.full(4, 1.0)
            lo = np.full(4, 0.0)
        elif 'joint_velocity' == key:
            hi = np.full(scene.arm_dof, 1.0)
            lo = -hi
        elif 'joint_position' == key:
            hi = np.full(scene.arm_dof, 3.14)
            lo = -hi
        elif 'grip_velocity' == key:
            hi = np.full(1, scene.max_gripper_velocity)
            lo = -hi
        elif 'grip_forces' == key:
            hi = np.full(2, scene.max_gripper_force)
            lo = np.full(2, 0)
        elif 'grip_width' == key:
            hi = 1.0
            lo = 0.0
        elif 'distance' in key:
            hi = np.ptp(scene.workspace, axis=0)
            lo = -hi
        elif 'position' in key:
            lo, hi = scene.workspace
        elif 'rgb' in key:
            lo = np.full(3, 0)
            hi = np.full(3, 255)
            dtype = np.uint8
        elif 'depth' in key:
            lo = 0
            hi = 255
            dtype = np.uint8
        elif 'mask' in key:
            lo = 0
            hi = 255
            dtype = np.uint8
        elif 'skill' in key:
            lo = -np.inf
            hi = np.inf
            dtype = np.uint8
        else:
            raise Exception("Unknown signal type: '{}'".format(key))

        if discrete:
            pass
        else:
            items[key] = spaces.Box(low=np.array(lo), high=np.array(hi),
                                    dtype=dtype)

    return spaces.Dict(items)
