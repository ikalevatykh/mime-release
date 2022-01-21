import click
import mime
import gym
import re

import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import InterpolationMode

from robos2r.core.tf import (
    pos_mat_from_vec,
    translation_from_matrix,
    quaternion_from_matrix,
)
from robos2r.data.lmdb import list_to_lmdb
from robos2r.config import data_path


def blend(im_a, im_b, alpha):
    im_a = Image.fromarray(im_a)
    im_b = Image.fromarray(im_b)
    im_blend = Image.blend(im_a, im_b, alpha).convert("RGB")
    im_blend = np.asanyarray(im_blend).copy()
    return im_blend


def data_process(crop_size):
    resize_im = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(crop_size),
        ]
    )
    resize_seg = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(crop_size, interpolation=InterpolationMode.NEAREST),
        ]
    )

    crop_transform = T.CenterCrop(crop_size)
    return resize_im, resize_seg, crop_transform


@click.command()
@click.option(
    "-p",
    "--path",
    default="sim2real/real/definitive/",
    type=str,
)
@click.option("-o", "--output_path", default="sim2real/real/stereo/", type=str)
@click.option("--crop_size", default=224, type=int)
@click.option("-db", "--debug/--no-debug", default=False, is_flag=True)
@click.option("-v", "--viz/--no-viz", default=False, is_flag=True)
def main(path, output_path, crop_size, debug, viz):

    directory = Path(data_path()) / path
    output_directory = Path(data_path()) / output_path
    output_directory.mkdir(parents=True, exist_ok=True)

    data_files = []
    data_files.extend(directory.glob("*.pkl"))

    # Define cleaning parameters
    cube_label = 3
    min_cube_area = 150
    gripper_label = 0
    min_gripper_area = 200

    resize_im, resize_seg, crop_transform = data_process(crop_size)

    dataset = []
    for data_file in tqdm(data_files):
        with open(str(data_file), "rb") as f:
            data = pkl.load(f)
            cam_list = data["cam_list"]
            try:
                dataset += data["dataset"]
            except:
                pass

    print(f"Processing dataset of size {len(dataset)}")

    env = gym.make("PRL_UR5-PickCamEnv-v0")
    env = env.unwrapped
    if debug:
        scene = env.unwrapped.scene
        scene.renders(True)

    clean_data_idx = []

    for scene_idx in tqdm(range(len(dataset))):
        scene = dataset[scene_idx]
        gripper_pose = scene["gripper_pose"]
        cube_pose = scene["target_position"], scene["target_orientation"]

        sim_obs = env.reset(
            cube_pose=cube_pose,
            gripper_pose=gripper_pose,
        )

        valid = True
        for cam_i in cam_list:

            cam_pos, cam_ori = scene[f"{cam_i}_optical_frame_tf"]
            match = re.search("(\d+)$", cam_i)

            cam_num = match.group(1)
            cam_name = cam_i[: -len(cam_num)]
            cam_num = int(cam_num)

            camera = env.cameras[cam_name][cam_num][0]
            camera.move_to(cam_pos, cam_ori)
            camera.shot()

            im = np.array(crop_transform(resize_im(np.flip(camera.rgb, (0, 1)))))
            seg = np.array(crop_transform(resize_seg(np.flip(camera.mask, (0, 1)))))
            im_real = np.array(
                crop_transform(resize_im(scene[f"rgb_{cam_i}"][:, :, :3]))
            )

            dataset[scene_idx][f"rgb_{cam_i}"] = im_real
            dataset[scene_idx][f"depth_{cam_i}"] = np.zeros((224, 224))

            if viz:
                im_blend = blend(im, im_real, 0.5)

                plt.subplot(121)
                plt.imshow(im_real)
                plt.subplot(122)
                plt.imshow(im)
                plt.show()
                plt.imshow(im_blend)
                plt.show()

            area_cube = np.sum(seg == cube_label)
            area_gripper = np.sum(seg == gripper_label)
            if area_cube < min_cube_area or area_gripper < min_gripper_area:
                valid = False
                break

        if valid:
            clean_data_idx.append(scene_idx)

    print(f"Clean dataset with {len(clean_data_idx)} samples")
    train_idx = np.random.choice(
        clean_data_idx, int(0.8 * len(clean_data_idx)), replace=False
    )
    val_idx = [i for i in clean_data_idx if i not in train_idx]

    print(f"Saving train dataset with {len(train_idx)} samples.")
    train_data = []
    for scene_idx in train_idx:
        train_data.append(dataset[scene_idx])
    train_path = str(output_directory / "train.lmdb")
    list_to_lmdb(train_data, train_path)

    print(f"Saving val dataset with {len(val_idx)} samples.")
    val_data = []
    for scene_idx in val_idx:
        val_data.append(dataset[scene_idx])

    val_path = str(output_directory / "val.lmdb")
    list_to_lmdb(val_data, val_path)


if __name__ == "__main__":
    main()
