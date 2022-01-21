import cv2
import time
import gym
import mime.envs
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt

from PIL import Image

import pickle as pkl

import torchvision.transforms as T

from robos2r.data.lmdb import LMDBDataset

from bulletman.envs.grasp import GraspNetEnv
from bulletman.envs.cube import CubeEnv
from bulletman.envs import PRLUR5Env, PRLCubeEnv


def main():
    with open("/home/rgarciap/Datasets/multi_cam.pkl", "rb") as f:
        d = pkl.load(f)

    resize_size = 480
    crop_size = 480

    crop_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(resize_size),
            T.CenterCrop(crop_size),
        ]
    )

    env = gym.make("PRL_UR5-PickCamEnv-v0")
    i = 0

    for scene in d:
        scene_data = scene["obs"]
        cube_pose = scene["cube_pose"]
        for i in range(len(scene_data)):
            print(f"Loading Scene data {i}")
            instance = scene_data[i]
            camera_pose = instance["camera_optical_frame_tf"]
            gripper_pose = instance["tool_tf"]
            obs = env.reset(
                cube_pose=cube_pose,
                # camera_pose=camera_pose,
                gripper_pose=gripper_pose,
                right_arm_idx=i,
            )
            im_real = crop_transform(instance["rgb"][:, :, :3])
            im = obs["rgb0"]

            print(f"Real Tool TF: {gripper_pose}")
            # print(obs)
            print(f"Sim Tool TF: {obs['tool_position'], obs['tool_orientation']}")

            plt.imshow(((im + im_real) / 2).astype(np.int))
            plt.show()
            plt.subplot(121)
            plt.imshow(im)
            plt.subplot(122)
            plt.imshow(im_real)
            plt.show()

            # print(f"Real Camera Pose: {camera_pose}")
            # print(f"Sim Camera Pose: {obs['camera_tf']}")
            # print(f"Sim Camera Pose: {obs['camera_tf']}")
            # print(f"Real Tool TF: {gripper_pose}")
            # print(f"Sim Tool TF: {obs['tool_tf']}")
            # print(f"Real Cube Pose {cube_pose}")
            # print(f"Sim Cube Pose {obs['cube_pose']}")


if __name__ == "__main__":
    main()
