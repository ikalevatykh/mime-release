import numpy as np
import pybullet as pb

from .body import Body
from .joint import JointArray


class Chain(JointArray):
    def __init__(self, body_id, tip_link_name, client_id):
        body = Body(body_id, client_id)
        tip = body.link(tip_link_name)

        joints = [i.info for i in body if not i.info.is_fixed]
        lowers = [i.lower_limit for i in joints]
        uppers = [i.upper_limit for i in joints]

        chain_indices = [i.index for i in tip.info.path if not i.is_fixed]
        chain_mask = [i.index in chain_indices for i in joints]

        super(Chain, self).__init__(body_id, chain_indices, client_id)
        self._tip = tip
        self._lowers = lowers
        self._uppers = uppers
        self._ranges = np.subtract(uppers, lowers)
        self._chain_mask = chain_mask

    @property
    def tip(self):
        return self._tip

    @property
    def limits(self):
        return (
            np.float32(self._lowers)[self._chain_mask],
            np.float32(self._uppers)[self._chain_mask],
        )
