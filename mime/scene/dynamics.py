import pybullet as pb


class Dynamics(object):
    def __init__(self, body_id, link_index, client_id):
        self._body_id = body_id
        self._link_index = link_index
        self.client_id = client_id

    def _get_info(self, index):
        info = pb.getDynamicsInfo(self._body_id, self._link_index,
                                  physicsClientId=self.client_id)
        return info[index]

    def _change(self, **kwargs):
        pb.changeDynamics(self._body_id, self._link_index,
                          physicsClientId=self.client_id, **kwargs)

    @property
    def mass(self):
        """ Mass in kg. """
        return self._get_info(0)

    @mass.setter
    def mass(self, value):
        self._change(mass=value)

    @property
    def lateral_friction(self):
        """ Lateral (linear) contact friction. """
        return self._get_info(1)

    @lateral_friction.setter
    def lateral_friction(self, value):
        self._change(lateralFriction=value)

    @property
    def rolling_friction(self):
        """ Torsional friction orthogonal to contact normal. """
        return self._get_info(6)

    @rolling_friction.setter
    def rolling_friction(self, value):
        self._change(rollingFriction=value)

    @property
    def spinning_friction(self):
        """ Torsional friction around the contact normal. """
        return self._get_info(7)

    @spinning_friction.setter
    def spinning_friction(self, value):
        self._change(spinningFriction=value)

    @property
    def contact_constraint(self):
        """ Stiffness and Damping (tuple) of the contact constraints
        for this body/link. This overrides the value if it was
        specified in the URDF file in the contact section. """
        return self._get_info(9), self._get_info(8)

    @contact_constraint.setter
    def contact_constraint(self, value):
        self._change(contactStiffness=value[0], contactDamping=value[1])

    @property
    def restitution(self):
        """  Bouncyness of contact. Keep it a bit less than 1. """
        return self._get_info(5)

    @restitution.setter
    def restitution(self, value):
        self._change(restitution=value)

    @property
    def local_inertia(self):
        """ Diagonal elements of the inertia tensor.
        Note that links and base are centered around the center of mass
        and aligned with the principal axes of inertia. """
        return self._get_info(2)

    @local_inertia.setter
    def local_inertia(self, value):
        self._change(localInertiaDiagnoal=value)

    @property
    def local_inertia_position(self):
        """ Position and orientation of inertial frame in local coordinates
        of the joint frame. """
        return (self._get_info(3), self._get_info(4))

    @property
    def linear_damping(self):
        """ Linear damping of the link (0.04 by default). """
        return NotImplementedError

    @linear_damping.setter
    def linear_damping(self, value):
        self._change(linearDamping=value)

    @property
    def angular_damping(self):
        """ Angular damping of the link (0.04 by default). """
        return NotImplementedError

    @angular_damping.setter
    def angular_damping(self, value):
        self._change(angularDamping=value)

    @property
    def friction_anchor(self):
        """ Enable or disable a friction anchor: positional friction correction.
        Disabled by default, unless set in the URDF contact section. """
        return None

    @friction_anchor.setter
    def friction_anchor(self, value):
        self._change(frictionAnchor=value)
