import collections

import numpy as np
import pybullet as pb

from . import collision
from .dynamics import Dynamics
from .joint import Joint, JointArray
from .link import Link
from .shape import VisualShape, CollisionShape
from .utils import augment_path


class Body(JointArray):
    def __init__(self, body_id, client_id, egl=False):
        self.client_id = client_id
        self.egl = egl
        num_joints = pb.getNumJoints(body_id, physicsClientId=client_id)
        super(Body, self).__init__(body_id, list(range(num_joints)), client_id)

    @staticmethod
    def num_bodies(client_id):
        return pb.getNumBodies(physicsClientId=client_id)

    @property
    def name(self):
        info = pb.getBodyInfo(self.body_id, physicsClientId=self.client_id)
        return info[-1].decode('utf8')

    def link(self, name):
        return next((Link(self.body_id, i.index, self.client_id)
                     for i in self.info if i.link_name == name), None)

    def joint(self, name):
        return next((Joint(self.body_id, i.index, self.client_id)
                     for i in self.info if i.joint_name == name), None)

    def joints(self, names):
        indices = []
        for name in names:
            ind = next((i.index for i in self.info if i.joint_name == name),
                       None)
            assert ind is not None, \
                'Unknown joint "{}" on body {} "{}"'.format(
                    name, self.body_id, self.name)
            indices.append(ind)
        return JointArray(self.body_id, indices, self.client_id)

    def links(self):
        return (Link(self.body_id, i) for i in self.joint_indices)

    def __getitem__(self, key):
        if isinstance(key, collections.Iterable):
            return JointArray(self._body_id, key, self.client_id)
        else:
            return super(Body, self).__getitem__(key)

    @property
    def position(self):
        return pb.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)

    @position.setter
    def position(self, pos_orn):
        if len(pos_orn) == 2:
            pos, orn = pos_orn
        else:
            pos, orn = pos_orn, (0, 0, 0, 1)
        if len(orn) == 3:
            orn = pb.getQuaternionFromEuler(orn)
        pb.resetBasePositionAndOrientation(
            self.body_id, pos, orn, physicsClientId=self.client_id)

    @property
    def color(self):
        return self.visual_shape.rgba_color

    @color.setter
    def color(self, value):
        if self.egl:
            # egl has a bug: when the color is set twice, it crashes
            return
        self.visual_shape.rgba_color = value

    @property
    def visual_shape(self):
        return VisualShape(self._body_id, -1, self.client_id)

    @property
    def collision_shape(self):
        return CollisionShape(self._body_id, -1, self.client_id)

    @property
    def dynamics(self):
        return Dynamics(self._body_id, -1, self.client_id)

    def get_overlapping_objects(self):
        """ Return all the unique ids of objects that have axis aligned
            bounding box overlap with a axis aligned bounding box of
            a given body. """
        return collision.get_overlapping_objects(self)

    def get_contacts(self, body_or_link_b=None):
        """ Returns the contact points computed during the most recent
            call to stepSimulation. """
        return collision.get_contact_points(self, body_or_link_b)

    def get_closest_points(self, max_distance, body_or_link_b=None):
        """ Compute the closest points, independent from stepSimulation.
            If the distance between objects exceeds this maximum distance,
            no points may be returned. """
        return collision.get_closest_points(self, body_or_link_b, max_distance)

    def get_collisions(self):
        """ Return all objects that intersect a given body. """
        return collision.get_collisions(self)

    def remove(self):
        pb.removeBody(self.body_id, physicsClientId=self.client_id)

    @staticmethod
    def load(file_name, client_id, egl=False, **kwargs):
        path = augment_path(file_name)
        loader = {
            '.urdf': pb.loadURDF,
            '.xml': pb.loadMJCF,
            '.sdf': pb.loadSDF
        }
        loader = loader[path.suffix.lower()]
        ids = loader(str(path), physicsClientId=client_id, **kwargs)
        if isinstance(ids, collections.Iterable):
            return [Body(i, client_id, egl) for i in ids]
        return Body(ids, client_id, egl)

    @staticmethod
    def create(visual_id,
               collision_id,
               client_id,
               pos=(0, 0, 0),
               orn=(0, 0, 0, 1),
               mass=0,
               egl=False):
        body_id = pb.createMultiBody(
            baseVisualShapeIndex=visual_id,
            baseCollisionShapeIndex=collision_id,
            basePosition=pos,
            baseOrientation=orn,
            baseMass=mass,
            physicsClientId=client_id)
        return Body(body_id, client_id, egl)

    @staticmethod
    def box(size, client_id, collision=False, mass=0, egl=False):
        size = np.array(size) / 2
        vis_id = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=size, physicsClientId=client_id)
        col_id = -1
        if collision:
            col_id = pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=size, physicsClientId=client_id)
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    @staticmethod
    def sphere(radius, client_id, collision=False, mass=0, egl=False):
        vis_id = pb.createVisualShape(
            pb.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
        col_id = -1
        if collision:
            col_id = pb.createCollisionShape(
                pb.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    @staticmethod
    def cylinder(radius, height, client_id, collision=False, mass=0, egl=False):
        vis_id = pb.createVisualShape(
            pb.GEOM_CYLINDER,
            radius=radius,
            length=height,
            physicsClientId=client_id)
        col_id = -1
        if collision:
            col_id = pb.createCollisionShape(
                pb.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=client_id)
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    @staticmethod
    def capsule(radius, height, client_id, collision=False, mass=0, egl=False):
        vis_id = pb.createVisualShape(
            pb.GEOM_CAPSULE,
            radius=radius,
            length=height,
            physicsClientId=client_id)
        col_id = -1
        if collision:
            col_id = pb.createCollisionShape(
                pb.GEOM_CAPSULE,
                radius=radius,
                height=height,
                physicsClientId=client_id)
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    @staticmethod
    def mesh(file_name, client_id, collision_file_name=None, scale=1, mass=0, egl=False):
        path = str(augment_path(file_name))
        if np.isscalar(scale):
            scale = (scale, ) * 3
        vis_id = pb.createVisualShape(
            pb.GEOM_MESH,
            fileName=path,
            meshScale=scale,
            physicsClientId=client_id)
        col_id = -1
        if collision_file_name is not None:
            path = str(augment_path(collision_file_name))
            col_id = pb.createCollisionShape(
                pb.GEOM_MESH,
                fileName=path,
                meshScale=scale,
                physicsClientId=client_id)
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    def __eq__(self, other):
        return self._body_id == other.body_id

    def __hash__(self):
        return self._body_id

    def __repr__(self):
        return 'Body({}) ""'.format(self._body_id, self.name)
