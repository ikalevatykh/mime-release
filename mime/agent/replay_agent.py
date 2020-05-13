import os
import click
import gym
import numpy as np
import pickle as pkl
from termcolor import colored


class ReplayAgent:
    def __init__(self, pickle_path):
        assert os.path.exists(pickle_path), 'File {} does not exist.'.format(pickle_path)
        self.dict_demos = pkl.load(open(pickle_path, 'rb'))
        self.demo = {}
        self.count = 0

    def set_seed(self, seed):
        assert seed in self.dict_demos['seed'],\
            'There is no recorded demonstration for seed {}.'.format(seed)
        self.demo = self.dict_demos['seed'][seed]
        self.count = 0

    def get_action(self):
        assert len(self.demo) > 0, 'Call set_seed first to load a demonstration.'
        action = None
        if self.count < len(self.demo):
            action = self.demo[self.count]
            self.count += 1
        return action


@click.command(help='replay_agent env_name [options]')
@click.argument('env_name', type=str)
@click.option('--seed', '-s', default=0, help='seed')
@click.option('--name', '-name', default='vr', help='name of the pickle file to load demo from')
def main(env_name, seed, name):
    env = gym.make(env_name)
    scene = env.unwrapped.scene
    scene.renders(True)
    env.seed(seed)
    obs = env.reset()

    agent = ReplayAgent(name+'.pkl')
    agent.set_seed(seed)

    done = False
    i = 0
    action = agent.get_action()
    while not done and action is not None:
        obs, reward, done, info = env.step(action)
        action = agent.get_action()

    if action is None:
        info['failure_message'] = 'End of Script.'

    if not info['success']:
        print(colored('Failure seed {}: {}'.format(seed, info['failure_message']), 'red'))
    else:
        print(colored('Success seed {}.'.format(seed), 'green'))


if __name__ == "__main__":
    main()
