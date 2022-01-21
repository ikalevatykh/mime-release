import cv2
import time
import gym
import mime.envs
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import pickle as pkl

import torchvision.transforms as T

from robos2r.data.lmdb import LMDBDataset

from bulletman.envs.grasp import GraspNetEnv
from bulletman.envs.cube import CubeEnv
from bulletman.envs import PRLUR5Env, PRLCubeEnv


def main():
    env = gym.make("DR-PRL_UR5-PickRandCamEnv-v0")
    i = 0
    scene = env.unwrapped.scene
    scene.renders(True)

    for i in tqdm(range(10000)):
        obs = env.reset()
        im = obs["rgb0"]
        seg = obs["mask0"]
        plt.imshow(im)
        plt.show()

        # plt.imshow(seg)
        # plt.show()

    # print(f"Real Camera Pose: {camera_pose}")
    # print(f"Sim Camera Pose: {obs['camera_tf']}")
    # print(f"Sim Camera Pose: {obs['camera_tf']}")
    # print(f"Real Tool TF: {gripper_pose}")
    # print(f"Sim Tool TF: {obs['tool_tf']}")
    # print(f"Real Cube Pose {cube_pose}")
    # print(f"Sim Cube Pose {obs['cube_pose']}")


if __name__ == "__main__":
    main()
