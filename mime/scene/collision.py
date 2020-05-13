import pybullet as pb

from . import body
from . import link


def get_contact_points(body_or_link_a, body_or_link_b=None):
    """ Returns the contact points computed during the most recent
        call to stepSimulation. """
    client_id = body_or_link_a.client_id
    kwargs = {}
    kwargs.update(_compute_args(body_or_link_a, 'bodyA', 'linkIndexA'))
    if body_or_link_b is not None:
        kwargs.update(_compute_args(body_or_link_b, 'bodyB', 'linkIndexB'))
    pts = pb.getContactPoints(physicsClientId=client_id, **kwargs)
    return [ContactPoint(p, body_or_link_a.client_id) for p in pts]


def get_closest_points(body_or_link_a, body_or_link_b=None, max_distance=10):
    """ Compute the closest points, independent from stepSimulation.
        If the distance between objects exceeds this maximum distance,
        no points may be returned. """

    client_id = body_or_link_a.client_id
    kwargs = {}
    kwargs.update(_compute_args(body_or_link_a, 'bodyA', 'linkIndexA'))
    if body_or_link_b is not None:
        kwargs.update(_compute_args(body_or_link_b, 'bodyB', 'linkIndexB'))
    pts = pb.getClosestPoints(distance=max_distance, physicsClientId=client_id, **kwargs)
    return [Distance(p, client_id) for p in pts]


def get_overlapping_objects(body_or_link):
    """ Return all the unique ids of objects that have axis aligned
        bounding box overlap with a axis aligned bounding box of
        given body/link. """
    client_id = body_or_link.client_id
    kwargs = _compute_args(body_or_link, 'bodyUniqueId', 'linkIndex')
    aa, bb = pb.getAABB(physicsClientId=client_id, **kwargs)
    obs = pb.getOverlappingObjects(aa, bb, physticsClient=client_id)
    if obs is None:
        return []
    return [link.Link(o[0], o[1]) for o in obs]


def get_collisions(body_or_link):
    """ Return all objects that intersect a given body/link. """

    # getOverlappingObjects doesnt work properly each time
    # overlap = get_overlapping_objects(link.Link(1,1))
    overlap = range(body.Body.num_bodies(body_or_link.client_id))
    return sum([get_closest_points(body_or_link, b, 0.0) for b in overlap
            if b != body_or_link.body_id], [])


def _compute_args(body_or_link, body_arg, link_arg):
    if isinstance(body_or_link, link.Link):
        return {body_arg: body_or_link.body_id,
                link_arg: body_or_link.link_index}
    elif isinstance(body_or_link, body.Body):
        return {body_arg: body_or_link.body_id}
    else:
        return {body_arg: body_or_link}


class Distance(object):
    def __init__(self, data, client_id):
        self._data = data
        self.client_id = client_id

    @property
    def body_a(self):
        """ Body A. """
        return body.Body(self._data[1], self.client_id)

    @property
    def body_b(self):
        """ Body B. """
        return body.Body(self._data[2], self.client_id)

    @property
    def link_a(self):
        """ Link of body A. """
        return link.Link(self._data[1], self._data[3], self.client_id)

    @property
    def link_b(self):
        """ Link of body B. """
        return link.Link(self._data[2], self._data[4], self.client_id)

    @property
    def position_on_a(self):
        """ Contact position on A, in Cartesian world coordinates (vec3). """
        return self._data[5]

    @property
    def position_on_b(self):
        """ Contact position on B, in Cartesian world coordinates (vec3). """
        return self._data[6]

    @property
    def distance(self):
        """ Distance, positive for separation, negative for penetration (float). """
        return self._data[8]

    def __repr__(self):
        return 'Distance {}-{}: {:.3}'.format(
            self.link_a, self.link_b, self.distance)


class ContactPoint(Distance):
    def __init__(self, data, client_id):
        super(ContactPoint, self).__init__(data, client_id)

    @property
    def contact_normal_on_b(self):
        """ Contact normal on B, pointing towards A (vec3). """
        return self._data[7]

    @property
    def normal_force(self):
        """ Normal force applied during the last 'stepSimulation' (float). """
        return self._data[9]
