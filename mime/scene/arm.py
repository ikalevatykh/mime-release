import numpy as np
import pybullet as pb

from .chain import Chain
from .controllable import Controllable
from .kinematics import DefaultKinematics


class Arm(Controllable):
    def __init__(self, body, tip_link_name):
        super(Arm, self).__init__()

        self._body = body
        self._chain = Chain(body.body_id, tip_link_name, body.client_id)
        self._kinematics = DefaultKinematics(self._chain, body.client_id)
        self._workspace = None

    def reset(self, joint_positions):
        """Instantly moves joints to a specific position.
        It is best only to do this at the start, while not running
        the simulation: it overrides all physics simulation.
        Args:
         joint_positions (vecN): Target joints position.
        """
        self._chain.reset(joint_positions)
        super(Arm, self).reset()

    def reset_tool(self, pos, orn=None):
        """Instantly moves tool to a specific cartesian position.
        Tries to solve IK for specified position and reset state of joints.
        It is best only to do this at the start, while not running
        the simulation: it overrides all physics simulation.
        Args:
         pos (vec3): Target position in Cartesian world coordinates.
         orn (vec4): Target orientation, quaternion (or euler angles).
                     If not set, current orientation will be used.
        Returns:
         True if succeed.
        """
        if orn is None:
            _, orn = self.tool.state.position
        elif len(orn) == 3:
            orn = pb.getQuaternionFromEuler(orn)

        q_init = self._chain.state.positions
        q_sol = self._kinematics.inverse_all(pos, orn, q_init)
        for q in q_sol:
            self.reset(q)
            if self.position_allowed:
                return True

        return False

    @property
    def joints(self):
        return self._chain

    @property
    def tool(self):
        return self._chain.tip

    @property
    def max_joint_velocity(self):
        return [i.max_velocity for i in self._chain.info]

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def joints_position(self):
        return self._chain.state.positions

    @property
    def tool_position(self):
        return self._chain.tip.state.position

    @property
    def position_allowed(self):
        """Check if joints is in limits and arm not in collision."""
        pos = self.joints_position
        low, high = self._chain.limits
        if np.any(pos < low) or np.any(pos > high):
            return False
        collisions = self._body.get_collisions()
        base_index = [0, 39]
        if [
            c for c in collisions if c.link_a.link_index not in base_index
        ]:  # except base link
            return False
        return True
