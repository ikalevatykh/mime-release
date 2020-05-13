import numpy as np
import pybullet as pb
from pyquaternion import Quaternion


class DefaultKinematics():
    def __init__(self, chain, client_id, damping=0.1):
        self._chain = chain
        self._damping = len(self._chain._lowers) * [damping]
        self.client_id = client_id

    def forward(self, q):
        raise NotImplementedError()

    def inverse(self, pos, orn, q_init=None):
        rest_poses = []
        if q_init is not None:
            rest_poses = [q_init]

        joint_pos = pb.calculateInverseKinematics(
            self._chain.body_id, self._chain.tip.link_index, pos, orn,
            jointDamping=self._damping,
            lowerLimits=self._chain._lowers, upperLimits=self._chain._uppers,
            jointRanges=self._chain._ranges, restPoses=rest_poses,
            physicsClientId=self.client_id)

        joint_pos = np.array(joint_pos)
        joint_pos[joint_pos > self._chain._uppers] -= 2 * np.pi
        joint_pos[joint_pos < self._chain._lowers] += 2 * np.pi
        return joint_pos[self._chain._chain_mask]

    def inverse_all(self, pos, orn, p_tol=1e-5, o_tol=1e-3, max_iter=100):
        rs = np.random.RandomState(777)  # deterministic randomization
        q_def = self._chain.state.positions
        q_sol = []
        for i in range(max_iter):
            q0 = rs.uniform(*self._chain.limits)
            self._chain.reset(q0)
            q = self.inverse(pos, orn, q_init=q0)

            p, o = self._chain.tip.state.position
            if np.allclose(p, pos, atol=p_tol) and \
                    _quatclose(o, orn, atol=o_tol):
                q_sol.append(q)

        self._chain.reset(q_def)
        return q_sol


def _quatclose(orn1, orn2, atol):
    """ Indicate quaternion similarities. It takes into account the fact
        that q and -q encode the same rotation. """
    q1 = Quaternion(orn1[3], *orn1[:3])
    q2 = Quaternion(orn2[3], *orn2[:3])
    return Quaternion.absolute_distance(q1, q2) < atol
