import itertools
import types
import numpy as np

import torch
import click
import gym
import time
import yaml

from robos2r.model import build_model
from .agent import Agent
from .script_agent import ScriptAgent, make_noised
from .utils import Rate
from PIL import Image
from pathlib import Path
from einops import rearrange
from torchvision import transforms as T


@click.command(help="script_agent env_name [options]")
@click.argument("env_name", type=str)
@click.option("-s", "--seed", default=0, help="seed")
@click.option("-t", "--times-repeat", default=1, help="times to repeat the script")
@click.option("-n", "--add-noise", is_flag=True, help="adding noise to actions or not")
@click.option(
    "-sc",
    "--skill-collection/--no-skill-collection",
    is_flag=True,
    help="whether to show the skills collection",
)
def main(env_name, seed, times_repeat, add_noise, skill_collection):
    print("Loading Augmentor model...")
    diffaug_model_path = "/home/rgarciap/Remote/models/diffs2r_new/resnet_adam_lr_1e-3_lraug0.01_bs_64_L8/"
    diffaug_model_path = Path(diffaug_model_path)
    diffaug_cfg_path = diffaug_model_path / "config.yml"

    with open(str(diffaug_cfg_path), "rb") as f:
        diffaug_cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_cfg = dict(
        name="diffaug",
        reg_output_size=3,
        aug_pipeline=diffaug_cfg["aug_pipeline"],
        multi=diffaug_cfg["multi_pipeline"],
        num_layers=diffaug_cfg["num_layers"],
        gumbel=diffaug_cfg["gumbel"],
        backbone_name=diffaug_cfg["backbone_name"],
    )
    diffaug_model = build_model(model_cfg)
    diffaug_ckp_path = diffaug_model_path / "best_checkpoint.pth"
    checkpoint = torch.load(str(diffaug_ckp_path), map_location="cpu")
    diffaug_model.load_state_dict(checkpoint["model"])
    augmentor = diffaug_model.augmentor
    augmentor.to("cpu")
    augmentor.eval()
    print("Model loaded")

    env = gym.make(env_name)
    scene = env.unwrapped.scene
    scene.renders(True)
    if skill_collection:
        scene.skill_data_collection = True
    env.seed(seed)
    for _ in range(times_repeat):
        obs = env.reset()

        agent = ScriptAgent(env)
        import matplotlib.pyplot as plt

        done = False
        i = 0
        rate = Rate(scene.dt)
        action = agent.get_action()
        if add_noise:
            make_noised(action)
        frames = []
        j = 0
        while not done and action is not None:
            obs, reward, done, info = env.step(action)

            im = T.ToTensor()(obs["rgb0"]).unsqueeze(0)
            mask = torch.tensor(obs["mask0"]).unsqueeze(0)

            im, mask = augmentor((im, mask))
            im = rearrange(im.detach().detach().squeeze(0).numpy(), "c h w -> h w c")
            im = Image.fromarray((im * 255).astype(np.uint8))
            im.save(f"0/output{j}.jpeg")
            j += 1
            action = agent.get_action()
            if add_noise and action is not None:
                make_noised(action)

        if action is None:
            info["failure_message"] = "End of Script."
        if not info["success"]:
            click.secho(
                "Failure Seed {}: {}".format(seed, info["failure_message"]), fg="red"
            )

        print("Success", info["success"])


if __name__ == "__main__":
    main()
