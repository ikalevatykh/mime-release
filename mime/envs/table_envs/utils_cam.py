from ...scene import Camera

import numpy as np
from itertools import product
from PIL import Image


def resize_kinect2(depth):
    off_w, off_h = (20, -20)
    depth = depth[off_h+62:off_h-20, off_w+65:off_w-105]
    depth = Image.fromarray(depth)
    depth = depth.resize((224, 224))
    depth = np.asarray(depth)
    return depth


def test_viewpoints(scene, cam_resolution, kn, kf):
    cam_params = scene.cam_params
    cam_rand = scene.cam_rand

    cam_keys = ['target', 'distance', 'yaw', 'pitch', 'fov', 'near', 'far']
    ref_params = [cam_params[key] for key in cam_keys]
    list_params = product(*[cam_rand[key] for key in cam_keys])

    obs_dic = {}
    camera = Camera(*cam_resolution, client_id=scene.client_id)
    for i, params in enumerate(list_params):
        params = [np.add(ref, p) for (ref, p) in zip(ref_params, params)]
        camera.view_at(target=params[0], distance=params[1],
                       yaw=params[2], pitch=params[3])
        camera.project(
            fov=params[4],
            near=params[5],
            far=params[6])
        camera.shot()
        obs_dic['rgb{}'.format(i)] = camera.rgb.copy()
        depth = camera.depth_uint8(kn=kn, kf=kf)
        # obs_dic['depth{}'.format(i)] = resize_kinect2(depth).copy()
        obs_dic['depth{}'.format(i)] = depth.copy()
        obs_dic['mask{}'.format(i)] = camera.mask.copy()

    return obs_dic
