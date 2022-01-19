import click
import gym

import matplotlib.pyplot as plt


@click.command()
@click.option("-e", "--env-name", default="PRL_UR5-EGL-Pick5RandCamEnv-v0", type=str)
def view(env_name):
    env = gym.make(env_name)
    obs = env.reset()
    print(obs.keys())
    for i in range(5):
        plt.imshow(obs[f"rgb{i}"])
        plt.show()
        plt.imshow(obs[f"depth{i}"])
        plt.show()
        plt.imshow(obs[f"mask{i}"])
        plt.show()


if __name__ == "__main__":
    view()
