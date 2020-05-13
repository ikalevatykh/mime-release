import os
import click
from joblib import Parallel, delayed
from PIL import Image
import json

def generate_images(env_name, seed, num_reset, images_dir):
    import mime
    import gym
    env = gym.make(env_name)
    for count_reset in range(num_reset):
        obs = env.reset()
        im_keys = ['depth',]
        for im_key in im_keys:
            count_cam = 0
            obs_key = '{}{}'.format(im_key, count_cam)
            while obs_key in obs:
                im = Image.fromarray(obs[obs_key])
                im_path = os.path.join(images_dir, '{:02}_{:02}_{}'.format(seed, count_reset, obs_key))
                cam_dict = env.unwrapped.cameras[count_cam][0].view_dict
                json.dump(cam_dict, open(im_path+'.json', 'w'))
                im.save(im_path+'.jpg', 'JPEG')
                count_cam += 1
                obs_key = '{}{}'.format(im_key, count_cam)

@click.command(help='script_agent env_name [options]')
@click.argument('env_name', type=str)
@click.option('-s', '--seed', default=0, help='seed')
@click.option('-n', '--num_processes', default=1, help='number of processes')
@click.option('-imdir', '--images_dir', default='images', help='path to save images')
@click.option('-n_im', '--num_images', default=20, help='number of images to generate')
def main(env_name, seed, num_images, images_dir, num_processes):
    assert 'RandCam' in env_name
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    Parallel(n_jobs=num_processes)(
        delayed(generate_images)(env_name, seed_worker, num_images//num_processes, images_dir)
        for seed_worker in range(seed, seed+num_processes))

if __name__ == "__main__":
    main()
