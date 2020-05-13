import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import click
import numpy as np
import struct
from io import BytesIO
from collections import OrderedDict

from .agent import Agent
from .utils import Rate

from ..net import conv_net
from ..net import attention_net
from ..net import resnet


class BernoulliTransform(object):
    def __call__(self, image):
        return image * torch.bernoulli(torch.ones(image.size()) * 0.8)


class NetAgent(Agent):
    def __init__(self, env, archi, net_path, timesteps, max_steps, epoch=-1, dim_action=4, steps_action=1):
        super(NetAgent, self).__init__(env)
        self._state_dim = 5
        self._timesteps = timesteps
        self._dim_action = dim_action
        self._tot_actions = dim_action * steps_action

        # multi GPU
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_gpu = torch.cuda.device_count()
        print('GPUs', num_gpu)

        # network architecture
        if archi == 'cnn':
            net = nn.DataParallel(conv_net.DepthNet(self._timesteps, self._tot_actions))
        elif archi == 'att':
            net = nn.DataParallel(attention_net.CNN(self._timesteps, self._tot_actions))
        elif archi == 'resnet18':
            net = nn.DataParallel(resnet.resnet18(pretrained=False, input_dim=timesteps, num_classes=self._tot_actions))
        elif archi == 'resnet50':
            net = nn.DataParallel(resnet.resnet50(pretrained=False, input_dim=timesteps, num_classes=self._tot_actions))
        elif archi == 'resnet152':
            net = nn.DataParallel(
                resnet.resnet152(pretrained=False, input_dim=timesteps, num_classes=self._tot_actions))
        elif archi == 'dann_resnet18':
            net = nn.DataParallel(resnet.dann_resnet18(pretrained=False, input_dim=timesteps, num_classes=self._tot_actions))

        # load snapshot
        if epoch == -1:
            suffix = 'current'
        else:
            suffix = str(epoch)
        path = '{}/{}_{}.pth'.format(net_path, archi, suffix)
        if str(self._device) == 'cpu':
            net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            net.load_state_dict(torch.load(path))
        net.eval()
        net.to(self._device)
        self._net = net

        # queues for image and signal history
        self._hist_im = None
        self._hist_sig = None

        self._count_steps = 0
        self._max_steps = max_steps

        # input image normalization
        self._transform = transforms.Compose([transforms.Resize([224, 224]),
                                              transforms.ToTensor(),
                                              # BernoulliTransform(),
                                              transforms.Normalize((0.5, 0.5), (0.5, 0.5)),
                                              ])

        # output control normalization in [-1, 1]
        self._factors = np.array([0.5, 0.5, 0.5, 4, 0.4, 0.4, 0.4])

    def get_image_from_bytes(self, raw_depth, max_depth_real):
        depth_header_size = 12
        header = raw_depth[:depth_header_size]
        [fmt, a, b] = struct.unpack('iff', header)
        raw_im = BytesIO(raw_depth[depth_header_size:])
        im_depth = Image.open(raw_im)
        im_depth = np.asarray(im_depth)
        im_depth_scaled = a / (im_depth.astype(np.float32) - b)
        im_depth_scaled[np.logical_or(im_depth == 0, im_depth_scaled > max_depth_real)] = 0
        im = im_depth_scaled / max_depth_real
        im_pil = Image.fromarray((im * 255).astype(np.uint8))

        return im_pil

    def get_action(self, raw_obs):

        self._count_steps += 1
        if -1 < self._max_steps < self._count_steps:
            return None

        obs_im = self.get_image_from_bytes(raw_obs, 1.8)
        obs_im = self._transform(obs_im)

        # queue of observed images
        skip = 1
        hist_im = self._hist_im
        if self._hist_im is None:
            hist_im = obs_im.repeat((self._timesteps * skip, 1, 1))
        else:
            # shift past images to the left
            hist_im[:-1, :, :] = hist_im[1:, :, :]
            hist_im[-1, :, :] = obs_im

        self._hist_im = hist_im
        obs_im_full = hist_im[::skip].float().to(self._device)

        # disable gradient computation at inference time
        with torch.no_grad():
            if 'dann' in self.archi:
                action, domain = self._net(obs_im_full[None, None, :], 0)[0]
            else:
                action = self._net(obs_im_full[None, None, :], 0)[0]

        dic_action = OrderedDict()

        action = action.cpu().numpy()
        if self._dim_action == 4:
            action = action[:4]
            action *= self._factors[3:]
            dic_action['grip_velocity'] = action[0]
            dic_action['linear_velocity'] = action[1:]
        else:
            action = action[:7]
            action *= self._factors
            dic_action['angular_velocity'] = action[:3]
            dic_action['grip_velocity'] = action[3]
            dic_action['linear_velocity'] = action[4:]

        if dic_action['grip_velocity'] < 0:
            dic_action['grip_velocity'] *= 2
        # elif 0 < dic_action['grip_velocity'] < 0.1:
        #     dic_action['grip_velocity'] = 0

        return dic_action


@click.command(help='net_agent env_name [options]')
@click.argument('env_name', type=str)
@click.option('--seed', default=0, help='seed')
@click.option('--archi', '-a', default='resnet18', help='net architecture')
@click.option('-np', '--net_path', type=str, required=True, help='net path')
@click.option('-ne', '--net_epoch', type=int, default=-1, help='net epoch to load')
@click.option('-da', '--dim_action', type=int, default=4, help='dimension of action output')
@click.option('-sa', '--steps_action', type=int, default=1, help='number of action steps output')
@click.option('-ts', '--timesteps', type=int, default=3, help='number of images to take as input')
def main(env_name, seed, archi, net_path, net_epoch, dim_action, steps_action, timesteps):
    import gym

    env = gym.make(env_name)
    scene = env.unwrapped.scene
    scene.renders(True)
    env.seed(seed)
    obs = env.reset()
    done = False

    agent = NetAgent(env, archi, net_path, timesteps, -1, net_epoch, dim_action, steps_action)
    action = agent.get_action(obs)
    rate = Rate(scene.dt)
    while not (action is None or done):
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        rate.sleep()

    if action is None:
        info['failure_message'] = 'End of Script.'
    if not info['success']:
        click.secho('Failure Seed {}: {}'.format(seed, info['failure_message']), fg='red')

    print('Success', info['success'])


if __name__ == "__main__":
    main()
