import numpy as np
import torchvision.transforms as T

from torchvision.transforms import InterpolationMode
from .table_env import TableEnv
from ...scene import Camera


class TableCamEnv(TableEnv):
    """Base environment for tasks with camera observation"""

    def __init__(
        self,
        scene,
        view_rand=False,
        gui_resolution=(640, 480),
        cam_resolution=(640, 480),
        num_cameras=1,
        crop_size=224,
    ):
        super(TableCamEnv, self).__init__(scene)
        scene.gui_resolution = gui_resolution

        self.observation_space = self._make_dict_space("rgb", "depth", "mask")

        self._view_rand = view_rand
        self._num_cameras = num_cameras
        self._cam_resolution = cam_resolution
        self._crop_size = crop_size

    def _reset(self, scene):
        np_random = self._np_random
        view_rand = self._view_rand
        num_cameras = self._num_cameras
        cam_resolution = self._cam_resolution

        robot_type = scene._robot_type

        cameras = []
        cam_params = scene.cam_params
        cam_rand = scene.cam_rand

        rand_params = {}
        if view_rand:
            for key in sorted(cam_rand.keys()):
                rand_range = cam_rand[key]
                size = 1
                if isinstance(cam_params[key], (list, tuple)):
                    size = len(cam_params[key])
                rand_offset = np_random.uniform(
                    rand_range[0], rand_range[1], (num_cameras, size)
                )
                rand_params[key] = cam_params[key] + rand_offset
        else:
            for key, rand_range in cam_rand.items():
                rand_params[key] = [cam_params[key]]

        cam_rand = scene.cam_rand
        if robot_type == "UR5":
            for target, dist, yaw, pitch, fov, near, far in zip(
                rand_params["target"],
                rand_params["distance"],
                rand_params["yaw"],
                rand_params["pitch"],
                rand_params["fov"],
                rand_params["near"],
                rand_params["far"],
            ):
                camera = Camera(*cam_resolution, client_id=self.scene.client_id)
                camera.project(fov=fov, near=near, far=far)
                camera.view_at(target=target, distance=dist, yaw=yaw, pitch=pitch)
                cameras.append((camera, near, far))
        elif robot_type == "PRL_UR5":
            fov, near, far = (
                cam_params["fov"],
                cam_params["near"],
                cam_params["far"],
            )
            for i in range(num_cameras):
                if len(scene.robot.wrist_cameras) < num_cameras:
                    w, h = self._cam_resolution
                    scene.robot.wrist_cameras.append(
                        scene.robot.attach_wrist_camera(width=w, height=h)
                    )
                camera = scene.robot.wrist_cameras[i]
                pos = rand_params["pos"][i]
                orn = rand_params["orn"][i]
                fov = rand_params["fov"][i]
                near = rand_params["near"][i]
                far = rand_params["far"][i]

                camera_link = camera._attach_link
                camera.attach(camera_link, pos=pos, orn=orn)
                camera.project(fov=fov, near=near, far=far)
                cameras.append((camera, near, far))

        self.cameras = cameras

        self._view_rand = view_rand
        self.num_cameras = num_cameras

    def _get_observation(self, scene):
        obs_dic = super(TableCamEnv, self)._get_observation(scene)

        resize_im = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self._crop_size),
            ]
        )
        resize_seg = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self._crop_size, interpolation=InterpolationMode.NEAREST),
            ]
        )

        crop_transform = T.CenterCrop(self._crop_size)

        for i, camera_nf in enumerate(self.cameras):
            camera, near, far = camera_nf
            camera.shot()
            rgb = camera.rgb.copy()
            depth = camera.depth_uint8(kn=near, kf=far)
            depth = depth.copy()
            mask = camera.mask.copy()

            obs_dic["rgb{}".format(i)] = np.array(crop_transform(resize_im(rgb)))
            # depth = camera.depth
            obs_dic["depth{}".format(i)] = np.array(crop_transform(resize_seg(depth)))
            obs_dic["mask{}".format(i)] = np.array(crop_transform(resize_seg(mask)))

        return obs_dic
