import click
import mime
import gym
import numpy as np
import time
import os

import torch.multiprocessing as mp

from tqdm import tqdm
from bulletman.envs.grasp import GraspNetEnv
from bulletman.envs.cube import CubeEnv
from bulletman.envs.prl_ur5 import PRLUR5Env
from bulletman.envs import PRLCubeEnv
from robos2r.data.lmdb import LMDBDataset, list_to_lmdb


def gathering_task(
    worker_id,
    env_name,
    randomize,
    num_scenes,
    task_queue,
):

    print(f"Env. name {env_name}")

    env = gym.make(env_name)
    cube_label = 2
    samples = []
    pbar = None
    if worker_id == 0:
        pbar = tqdm(total=num_scenes, ncols=80)

    for i in range(num_scenes):
        valid = False
        while not valid:
            obs = env.reset()
            seg = obs["mask0"]
            area = np.sum(seg == cube_label)
            valid = area > 150

        samples.append(obs)
        if pbar is not None:
            pbar.update()
    if pbar is not None:
        pbar.close()
    task_queue.put((worker_id, samples))
    print(f"Worker {worker_id} finished.")
    del samples


@click.command()
@click.option("-o", "--output-path")
@click.option("-e", "--env-name", default="PRL_UR5-PickRandCamEnv-v0", type=str)
@click.option("-ns", "--num-scenes", default=40000, type=int)
@click.option("-nw", "--num-workers", default=1, type=int)
@click.option("-randomize", "--randomize/--no-randomize", default=False, is_flag=True)
def collect(
    output_path,
    env_name,
    num_scenes,
    num_workers,
    randomize,
    egl,
):
    print(f"Write to {output_path}")
    scenes_per_worker = num_scenes // num_workers
    scenes_last_worker = num_scenes - scenes_per_worker * (num_workers - 1)

    task_queue = mp.Queue()

    output_path = os.path.expandvars(output_path)

    workers = []

    for i in range(num_workers):
        worker = mp.Process(
            target=gathering_task,
            args=(
                i,
                env_name,
                randomize,
                scenes_per_worker if i < num_workers - 1 else scenes_last_worker,
                task_queue,
            ),
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    finished = 0
    samples = []
    while finished < num_workers:
        worker_id, data = task_queue.get()
        samples += data
        print(f"Worker {worker_id} finished the task.")
        finished += 1

    list_to_lmdb(samples, output_path)


if __name__ == "__main__":
    collect()
