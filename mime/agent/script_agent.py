import itertools
import types
import numpy as np

import click
import gym
import time

from .agent import Agent
from .utils import Rate


class ScriptAgent(Agent):
    def __init__(self, env):
        super(ScriptAgent, self).__init__(env)
        self._scripts = env.unwrapped.scene.script()
        if not isinstance(self._scripts, types.GeneratorType):
            self._multiple_scripts = False
            self._steps = itertools.chain(*self._scripts)
        else:
            self._multiple_scripts = True
            self._steps = itertools.chain([])

    def _compute_script(self):
        script = next(self._scripts, None)
        if script is None:
            self._steps = itertools.chain([])
        else:
            self._steps = itertools.chain(*script)

    """Pre-compute actions at each new script"""

    def get_action_update(self):
        action = next(self._steps, None)
        if action is None and self._multiple_scripts:
            self._compute_script()
            action = next(self._steps, None)

        return action


def make_noised(action):
    if "joint_velocity" in action:
        action["joint_velocity"] += np.random.normal(scale=0.01, size=6)
    if "linear_velocity" in action:
        action["linear_velocity"] += np.random.normal(scale=0.007, size=3)
    if "angular_velocity" in action:
        action["angular_velocity"] += np.random.normal(scale=0.04, size=3)


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
    env = gym.make(env_name)
    scene = env.unwrapped.scene
    scene.renders(True)
    if skill_collection:
        scene.skill_data_collection = True
    env.seed(seed)
    for _ in range(times_repeat):
        obs = env.reset()

        agent = ScriptAgent(env)

        done = False
        i = 0
        rate = Rate(scene.dt)
        action = agent.get_action()
        if add_noise:
            make_noised(action)
        frames = []

        while not done and action is not None:
            print(action)
            print(obs["grip_velocity"])
            obs, reward, done, info = env.step(action)
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
