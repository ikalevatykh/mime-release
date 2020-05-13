import numpy as np
import pybullet as pb
from pyquaternion import Quaternion

from . import link
from . import body


class Constraint(object):
    def __init__(self, constraint_id, client_id):
        self._constraint_id = constraint_id
        self.client_id = client_id

    @staticmethod
    def create_fixed(link_a, link_b=None):
        """ Create a fixed constraint between objects,
        or between an object and a specific world frame.
        Args:
            link_a: Parent link or body.
            link_b: Child link or body.
                    If not set, non-dynamic world frame will be used.
        """

        parent_body_id = link_a.body_id
        parent_link_id = link_a.link_index
        parent_position = link_a.state.position

        if isinstance(link_b, body.Body):
            child_body_id = link_b.body_id
            child_link_id = -1
            child_position = link_b.position
        elif isinstance(link_b, link.Link):
            child_body_id = link_b.body_id
            child_link_id = link_b.link_index
            if child_link_id == -1:
                child_position = body.Body(link_b.body_id, link_b.client_id).position
            else:
                child_position = link_b.state.position
        else:
            child_body_id = -1
            child_link_id = -1
            child_position = (0, 0, 0), (0, 0, 0, 1)

        base_pos, base_orn = tf(parent_position)
        child_pos, child_orn = tf(child_position)
        pos = base_orn.inverse.rotate(child_pos - base_pos)
        orn = base_orn.inverse * child_orn
        orn = [*orn.vector, orn.scalar]

        id = pb.createConstraint(
            parent_body_id, parent_link_id,
            child_body_id, child_link_id,
            pb.JOINT_FIXED,
            (0, 0, 0),
            parentFramePosition=pos,
            childFramePosition=(0, 0, 0),
            parentFrameOrientation=orn,
            childFrameOrientation=(0, 0, 0, 1),
            physicsClientId=link_a.client_id
        )
        return Constraint(id, link_a.client_id)

    def remove(self):
        pb.removeConstraint(self._constraint_id, physicsClientId=self.client_id)


def tf(position):
    pos, orn = position
    pos = np.array(pos)
    orn = Quaternion(orn[3], *orn[:3])
    return pos, orn
