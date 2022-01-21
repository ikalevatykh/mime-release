import pybullet as pb
import pickle as pkl
import numpy as np
import torch
import torchvision.transforms as T
import time
from einops import rearrange

from mime.config import assets_path

from torchvision.transforms import InterpolationMode
from .table_env import TableEnv
from ...scene import Camera


class TableCamEnv(TableEnv):
    """Base environment for tasks with camera observation"""

    def __init__(
        self,
        scene,
        view_rand="",
        gui_resolution=(640, 480),
        cam_resolution=(1280, 720),
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

        self.done_move = False

        self.cameras = {}

        self.cam_list = []

        camera_cfgs = scene.camera_cfgs
        self.cam_list = [
            f"{cam_info['name']}{i}"
            for cam_info in camera_cfgs
            for i in range(num_cameras)
        ]

        self.joints_pos = {}

        self.resize_crop_im = T.Compose(
            [
                T.Resize(self._crop_size, antialias=True),
                T.CenterCrop(self._crop_size),
            ]
        )
        self.resize_crop_seg = T.Compose(
            [
                T.Resize(self._crop_size, interpolation=InterpolationMode.NEAREST),
                T.CenterCrop(self._crop_size),
            ]
        )

    def _reset(self, scene):
        np_random = self._np_random
        view_rand = self._view_rand
        num_cameras = self._num_cameras
        cam_resolution = self._cam_resolution

        robot_type = scene._robot_type
        camera_cfgs = scene.camera_cfgs

        if scene._domain_rand:
            (
                light_color,
                light_distance,
                light_direction,
                diffuse_coeff,
                specular_coeff,
                ambient_coeff,
                shadows,
            ) = scene._modder.randomize_lighting(np_random)
        else:
            (
                light_color,
                light_distance,
                light_direction,
                diffuse_coeff,
                specular_coeff,
                ambient_coeff,
                shadows,
            ) = scene._modder.randomize_lighting(np_random)

        for cam in camera_cfgs:
            cam_name = cam["name"]
            def_system = cam["def_system"]
            ext_params = cam["ext_params"]
            rand_params = cam["rand_params"]
            attached = cam["attached"]
            ext_rand_params = {}

            if cam_name not in self.cameras.keys():
                self.cameras[cam_name] = []
            if cam_name not in self.joints_pos.keys():
                self.joints_pos[cam_name] = []

            if def_system == "cartesian":
                if view_rand:
                    for key in sorted(rand_params.keys()):
                        rand_range = rand_params[key]
                        size = 1
                        if isinstance(ext_params[key], (list, tuple)):
                            size = len(ext_params[key])

                        if key == "rotation":
                            size = 3
                        rand_offset = np_random.uniform(
                            rand_range[0], rand_range[1], (num_cameras, size)
                        )

                        if key == "rotation":
                            rot_quat = ext_params[key]
                            rot_euler = pb.getEulerFromQuaternion(rot_quat)
                            ext_rand_params[key] = rot_euler + rand_offset
                        else:
                            ext_rand_params[key] = ext_params[key] + rand_offset

                else:
                    for key, rand_range in rand_params.items():
                        ext_rand_params[key] = [ext_params[key]]

                params = list(
                    zip(
                        ext_rand_params["translation"],
                        ext_rand_params["rotation"],
                        ext_rand_params["fov"],
                        ext_rand_params["near"],
                        ext_rand_params["far"],
                    )
                )

                for i in range(num_cameras):
                    cam_pos, cam_orn, fov, near, far = params[i]

                    if len(cam_orn) == 3:
                        cam_orn = pb.getQuaternionFromEuler(cam_orn)

                    w, h = self._cam_resolution
                    if attached:
                        if len(scene.robot.wrist_cameras) < i + 1:
                            scene.robot.wrist_cameras.append(
                                scene.robot.attach_wrist_camera(width=w, height=h)
                            )
                            self.joints_pos[cam_name].append([0, 0, 0, 0, 0, 0])
                        camera = scene.robot.wrist_cameras[i]

                        q = pb.calculateInverseKinematics(
                            scene.robot._body.body_id,
                            scene.robot._body.link(
                                "right_camera_color_optical_frame"
                            ).link_index,
                            targetPosition=cam_pos,
                            targetOrientation=cam_orn,
                        )[14:20]

                        self.joints_pos[cam_name][i] = q

                    else:
                        if len(self.cameras[cam_name]) < i + 1:
                            camera = Camera(w, h, client_id=self.scene.client_id)
                        else:
                            camera, _, _, _, _, _ = self.cameras[cam_name][i]

                    camera.set_lighting(
                        light_color,
                        light_distance,
                        light_direction,
                        diffuse_coeff,
                        specular_coeff,
                        ambient_coeff,
                        shadows,
                    )

                    camera.project(fov=fov, near=near, far=far)
                    camera.move_to(cam_pos, cam_orn)

                    if len(self.cameras[cam_name]) < i + 1:
                        self.cameras[cam_name].append(
                            (
                                camera,
                                near,
                                far,
                                attached,
                                cam_pos,
                                cam_orn,
                            )
                        )
                    else:
                        self.cameras[cam_name][i] = (
                            camera,
                            near,
                            far,
                            attached,
                            cam_pos,
                            cam_orn,
                        )
            else:
                raise ValueError(f"Def system {def_system} is not valid.")

    def _get_observation(self, scene):
        obs_dic = super(TableCamEnv, self)._get_observation(scene)

        cam_dic = dict(rgb=[], depth=[], mask=[])
        cam_names = []
        for cam_name, cameras_list in self.cameras.items():
            for i, camera_info in enumerate(cameras_list):
                camera, near, far, attached, cam_pos, cam_orn = camera_info

                if attached:
                    q = self.joints_pos[cam_name][i]
                    scene.robot.right_arm.reset(q)

                    cam_pos, cam_orn = self._scene.robot.right_arm.tool_position
                    obs_dic[f"camera_optical_frame_tf_{cam_name}{i}"] = (
                        cam_pos,
                        cam_orn,
                    )
                else:
                    obs_dic[f"camera_optical_frame_tf_{cam_name}{i}"] = (
                        cam_pos,
                        cam_orn,
                    )

                camera.shot()
                rgb = torch.tensor(camera.rgb)
                depth = torch.tensor(camera.depth_uint8(kn=near, kf=far))
                mask = torch.tensor(camera.mask)
                if not attached:
                    rgb = torch.flip(rgb, (0, 1))
                    depth = torch.flip(depth, (0, 1))
                    mask = torch.flip(mask, (0, 1))

                cam_dic["rgb"].append(rgb)
                cam_dic["depth"].append(depth)
                cam_dic["mask"].append(mask)
                cam_names.append(f"{cam_name}{i}")

        for channel, v in cam_dic.items():
            v_new = torch.stack(v).float()
            if channel == "rgb":
                v_new = rearrange(v_new, "b h w c -> b c h w")
                v_new = self.resize_crop_im(v_new)
                v_new = rearrange(v_new, "b c h w -> b h w c")
            else:
                # assumes other modalities are of the form (b h w)
                v_new = self.resize_crop_seg(v_new)
            v_new = np.asarray(v_new, dtype=np.uint8)
            for i, cam_name in enumerate(cam_names):
                obs_dic[f"{channel}_{cam_name}"] = v_new[i]

        return obs_dic
