import os
import click
import gym
import numpy as np
import pickle as pkl
from termcolor import colored

from mime.agent.vr_agent import VRAgent


def load_pkl(path, env_name, range_demos):
    if os.path.exists(path):
        dict_demos = pkl.load(open(path, 'rb'))
        assert dict_demos['env'] == env_name
        for key in dict_demos['seed'].keys():
            if key in range_demos:
                print('Seeds in dataset: ', dict_demos['seed'].keys())
                raise ValueError('A demonstration for seed {} is already present in the dataset.'.format(key))
    else:
        dict_demos = {'env': env_name, 'seed': {}}
    return dict_demos


def collect(env, seed, timescale):
    env.seed(seed)
    env.reset()
    agent = VRAgent(env, timescale)
    done = False
    actions = []
    while not done:
        action = agent.get_action()
        if action is not None:
            obs, reward, done, info = env.step(action)
            actions.append(action)
    return actions, info


@click.command(help='VR data collection')
@click.argument('env_name', type=str)
@click.option('--seed', '-s', default=0, help='start seed')
@click.option('--episodes', '-e', default=1, help='start seed')
@click.option('--name', '-name', default='vr', help='name of the pickle file to save demos in')
@click.option('-t', '--timescale', type=int, default=10, help='time scale')
def main(env_name, seed, episodes, name, timescale):
    env = gym.make(env_name).unwrapped
    env.observe(False)
    env.renders(shared=True)

    range_demos = list(range(seed, seed+episodes))
    path_pkl = name+'.pkl'
    dict_demos = load_pkl(path_pkl, env_name, range_demos)
    dict_demos['timescale'] = timescale

    for current_seed in range_demos:
        actions, info = collect(env, current_seed, timescale)
        if info['success']:
            dict_demos['seed'][current_seed] = actions
            pkl.dump(dict_demos, open(path_pkl, 'wb'))
            print(colored('Demonstration for seed {} added to the dataset.'.format(current_seed), 'green'))
        else:
            print(colored('Demonstration for seed {} was a failure. Not recorded.'.format(current_seed), 'red'))

if __name__ == '__main__':
    main()
