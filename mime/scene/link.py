import pybullet as pb

from . import collision
from .dynamics import Dynamics
from .joint import JointInfo, Joint
from .shape import VisualShape, CollisionShape


class Link(object):
    def __init__(self, body_id, link_index, client_id):
        self._body_id = body_id
        self._link_index = link_index
        self.client_id = client_id

    @property
    def body_id(self):
        return self._body_id

    @property
    def link_index(self):
        return self._link_index

    @property
    def info(self):
        return JointInfo(self._body_id, self._link_index, self.client_id)

    @property
    def parent_joint(self):
        return Joint(self._body_id, self._link_index, self.client_id)

    @property
    def state(self):
        return LinkState(self._body_id, self._link_index, self.client_id)

    @property
    def visual_shape(self):
        return VisualShape(self._body_id, self._link_index, self.client_id)

    @property
    def collision_shape(self):
        return CollisionShape(self._body_id, self._link_index, self.client_id)

    @property
    def dynamics(self):
        return Dynamics(self._body_id, self._link_index, self.client_id)

    def get_overlapping_objects(self):
        """ Return all the unique ids of objects that have axis aligned
            bounding box overlap with a axis aligned bounding box of
            a given link. """
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

    def __eq__(self, other):
        return self._body_id == other.body_id and \
               self._link_index == other.link_index

    def __hash__(self):
        return self._body_id + self._link_index << 24

    def __repr__(self):
        return 'Link({}:{})'.format(self._body_id, self._link_index)


class LinkState(object):
    def __init__(self, body_id, joint_index, client_id):
        self._body_id = body_id
        self._joint_index = joint_index
        self._compute_velocity = 0
        self._state = pb.getLinkState(
            body_id, joint_index, 0, physicsClientId=client_id)
        self.client_id = client_id

    @property
    def position(self):
        """ Cartesian position of center of mass """
        return self._state[0], self._state[1]

    @property
    def local_inertial_frame_position(self):
        return self._state[2], self._state[3]

    @property
    def world_link_frame_position(self):
        ''' Position of URDF link '''
        return self._state[4], self._state[5]

    @property
    def velocity(self):
        ''' World link velocity '''
        if not self._compute_velocity:
            self._state = pb.getLinkState(
                self._body_id,
                self._joint_index,
                1,
                physicsClientId=self.client_id)
            self._compute_velocity = True
        return self._state[6], self._state[7]
