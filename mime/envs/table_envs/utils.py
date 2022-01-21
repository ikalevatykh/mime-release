import math
import os
import numpy as np
import json
import random

import pybullet as pb

from tqdm import tqdm
from mime.scene import Body
from mime.settings import SHAPENET_PATH


class MeshLoader:
    def __init__(
        self,
        client_id,
        folders,
        ratios=(1,),
        train=True,
        simplified=False,
        verbose=False,
        egl=False,
        reduced_radius_model=False,
    ):
        self.client_id = client_id
        self.simplified = simplified
        self.reduced_radius_model = reduced_radius_model
        self.root = SHAPENET_PATH
        self.egl = egl
        self.folders = folders
        self.files = []
        self.dic_labels = json.load(open(os.path.join(self.root, "labels.json"), "r"))

        for folder, ratio in zip(folders, ratios):
            self.files_folder = [
                os.path.join(folder, file)
                for file in self.dic_labels[folder]["train"]
                + self.dic_labels[folder]["test"]
            ]
            self.files_folder = self.files_folder[: int(ratio * len(self.files_folder))]
            self.files += self.files_folder

        self.verbose = verbose
        if self.verbose:
            print("Folder {} Num {}".format(folder, len(self.files)))

    def __len__(self):
        return len(self.files)

    def get_mesh(self, idx, scale, reduced_radius_model=False, useFixedBase=False):
        if self.simplified and os.path.exists(
            os.path.join(self.root, self.files[idx], "models", "model_simplified.urdf")
        ):
            if self.reduced_radius_model and os.path.exists(
                os.path.join(
                    self.root, self.files[idx], "models", "model_simplified_radius.urdf"
                )
            ):
                urdf = "model_simplified_radius.urdf"
            else:
                urdf = "model_simplified.urdf"
        else:
            urdf = "model_normalized.urdf"
        path_mesh = os.path.join(self.root, self.files[idx], "models", urdf)

        if self.verbose:
            print("Loading mesh", path_mesh)

        mesh = Body.load(
            path_mesh,
            client_id=self.client_id,
            globalScaling=scale,
            flags=pb.URDF_USE_IMPLICIT_CYLINDER,
            egl=self.egl,
            useFixedBase=useFixedBase,
        )
        return mesh


def aabb_collision(b1, b2):
    no_collision = False
    for i in range(3):
        no_collision = no_collision or b2[1][i] < b1[0][i] or b1[1][i] < b2[0][i]

    return not no_collision


def sample_without_overlap(
    mesh,
    aabbs,
    np_random,
    low,
    high,
    low_z_rot=0,
    high_z_rot=0,
    x_rot=0,
    min_dist=0,
    verbose=False,
    max_num_trials=None,
):
    mesh_pos = np.zeros(3)
    mesh_z_rot = np_random.uniform(low_z_rot, high_z_rot)
    mesh.position = mesh.position[0], [x_rot, 0, mesh_z_rot]
    mesh_aabb = mesh.collision_shape.AABB
    mesh_pos[2] = (mesh_aabb[1][2] - mesh_aabb[0][2]) / 2  # z coord

    count_sample = 0
    overlap = True
    num_trials = max_num_trials if max_num_trials is not None else 10 ** 5
    while overlap and count_sample < num_trials:
        mesh_pos[:2] = np_random.uniform(low[:2], high[:2])
        mesh.position = mesh_pos, [x_rot, 0, mesh_z_rot]
        mesh_aabb = np.array(mesh.collision_shape.AABB)
        mesh_aabb[0] -= min_dist
        mesh_aabb[1] += min_dist
        if verbose:
            print("mesh aabb", mesh_aabb)
        overlap = any([aabb_collision(mesh_aabb, aabb) for aabb in aabbs])
        count_sample += 1

    if overlap and max_num_trials is not None:
        # could not find a place for the object
        return None, mesh_z_rot

    aabbs.append(mesh_aabb)
    return aabbs, mesh_z_rot


def conf_to_radians(joint_values):
    return {k: math.radians(v) for k, v in joint_values.items()}


def load_textures(path, np_random, max_number=None):
    textures = []
    print(path)
    texture_paths = list(path.glob("*.jpg"))
    np_random.shuffle(texture_paths)

    textures_len = len(texture_paths)

    if max_number is not None:
        textures_len = min(textures_len, max_number)

    print(f"Loading {textures_len} textures for {path.name}.")
    for i in tqdm(range(textures_len)):
        texture_path = texture_paths[i]
        texture_id = pb.loadTexture(str(texture_path))
        textures.append(texture_id)
    return textures
