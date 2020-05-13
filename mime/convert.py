import click
import json
import os

import numpy as np
import pybullet as pb
from tqdm import tqdm
import xml.etree.ElementTree as ET


def add_cylinder(collision, xyz, radius, length):
    geometry = ET.SubElement(collision, 'geometry')
    cylinder = ET.SubElement(geometry, 'cylinder')
    cylinder.set('radius', str(radius))
    cylinder.set('length', str(length))

    origin = ET.SubElement(collision, 'origin')
    origin.set('rpy', '-1.57 0 0')
    origin.set('xyz', '{} {} {}'.format(*xyz))


def add_box(collision, xyz, size):
    geometry = ET.SubElement(collision, 'geometry')
    box = ET.SubElement(geometry, 'box')
    box.set('size', '{} {} {}'.format(*size))

    origin = ET.SubElement(collision, 'origin')
    origin.set('rpy', '0 0 0')
    origin.set('xyz', '{} {} {}'.format(*xyz))


@click.command(help='convert label [options]')
@click.argument('label', type=str)
@click.option('-r', '--root', help='shapenet root path')
@click.option('-c', '--check', default=False, help='check output urdfs')
def convert(label, root, check):
    if label == 'mug':
        folder = '03797390'
    elif label == 'bottle':
        folder = '02876657'
    else:
        raise RuntimeError('Unknown label: {}'.format(label))

    dic_labels = json.load(
        open(os.path.join(root, 'labels.json'), 'r'))

    files = [
        os.path.join(folder, file)
        for dataset in ['train', 'test', 'excluded']
        for file in dic_labels[folder][dataset]
    ]

    pb.connect(pb.DIRECT)

    pbar = tqdm(files)
    for file in pbar:
        pbar.set_description(file)

        urdf = os.path.join(root, file, 'models/model_normalized.urdf')
        urdf_out = os.path.join(root, file, 'models/model_simplified.urdf')

        pb.resetSimulation()
        i = pb.loadURDF(urdf)
        aabb = pb.getAABB(i)

        xyz = np.mean(np.array(aabb), axis=0)
        dim = np.ptp(np.array(aabb), axis=0)

        tree = ET.parse(urdf)
        collision = tree.getroot().findall('.//collision')[0]

        # remove old collision
        geometry = collision.find('geometry')
        collision.remove(geometry)

        # add new collision geometry
        if label == 'bottle':
            if np.abs(dim[0] - dim[2]) < 0.1:
                radius = np.max([dim[0], dim[2]]) / 2.0
                length = dim[1]
                add_cylinder(collision, xyz, radius, length)
            else:
                add_box(collision, xyz, dim)
        elif label == 'mug':
            handle = dim[2] - dim[0]
            origin = xyz
            origin[2] += handle / 2
            radius = dim[0] / 2.0
            length = dim[1]
            add_cylinder(collision, origin, radius, length)

        tree.write(urdf_out)

        if check:
            pb.resetSimulation()

            i = pb.loadURDF(urdf_out, flags=pb.URDF_USE_IMPLICIT_CYLINDER)
            aabb = pb.getAABB(i)

            xyz_out = np.median(np.array(aabb), axis=0)
            dim_out = np.ptp(np.array(aabb), axis=0)

            if label == 'bottle':
                if not np.allclose(xyz, xyz_out, atol=0.1) or \
                        not np.allclose(dim, dim_out, atol=0.1):
                    raise RuntimeError('conversion failed')


if __name__ == "__main__":
    convert()
