## How to set up the camera parameters

The camera parameters are different for each setup (`default`, `paris` and `grenoble`). They are located in `mime/envs/table_envs/table_scene.py`:
```
elif self._setup == 'grenoble':
    self.cam_params = {
        'target': (0, 0, 0.2),
        'distance': 1.26,
        'yaw': 90,
        'pitch': -20,
        'fov': 60,
        'near': 0.3,
        'far': 1.8
    }
    self.cam_rand = {
        'target': (-0.02, 0.02),
        'distance': (0, 0.1),
        'yaw': (-20, 20),
        'pitch': (-8, 4),
        'fov': (-5, 5),
        'near': (-0.05, 0.05),
        'far': (-0.05, 0.05)
    }
```

`cam_params` have to match the real world camera. First, make sure that your camera is well centered with respect to the robot base (normally, the depth sensors are located not exactly in the center of the camera itself). Then, you need to measure the real world values of:
1) `target`: the position in the space where the camera looks at. It might be the robot base `(0, 0, 0)` or a position above it (with different z value);
2) `distance`: distance between the target position and the camera;
3) `pitch`: the angle of the camera with respect to the target position xy plane.

We use squared images in mime while real world images have different hight and width. For example, Kinect2 has vertical field of view of 60 degrees and horizontal field of view of 71 degrees which corresponds to an image of 424x512 pixels. To find the matching between sim and real, crop the image to be a square (424x424) and use the vertical field of view in sim (`fov = 60`). The `yaw` parameter should be always `90` if the camera is well centered. `near` and `far` parameters should correspond to the depth postprocessing parameters used while recording the real world dataset.

`cam_rand` have to guarantee that the object in any part of the workspace if visible from any random camera position. You can something like this to test the camera randomization:

```
import gym
import mime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mime.envs.table_envs.utils_cam import test_viewpoints


def main(num_trials):
    env_name = 'UR5-PickCamEnv-v0'
    env = gym.make(env_name)
    cube_mask_value = 3
    env.seed(5)
    for trial in range(num_trials):
        env.reset()
        viewpoints = test_viewpoints(env.unwrapped.scene, (224, 224), 0.3, 1.8)
        num_cameras = len(viewpoints)//3
        for camera in range(num_cameras):
            mask = viewpoints['mask{}'.format(camera)]
            if cube_mask_value not in mask:
                plt.imshow(mask)
                plt.show()
        print('Trial {} is successful'.format(trial))


if __name__ == '__main__':
    main(num_trials=100)
```
