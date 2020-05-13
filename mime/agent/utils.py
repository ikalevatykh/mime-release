import time
import numpy as np
from pyquaternion import Quaternion


def tf(position):
    pos, orn = position
    pos = np.array(pos)
    if not isinstance(orn, Quaternion):
        orn = Quaternion(orn[3], *orn[:3])
    return pos, orn


class KinematicConstraint(object):
    def __init__(self, base_position, child_position):
        base_pos, base_orn = tf(base_position)
        child_pos, child_orn = tf(child_position)
        pos = base_orn.inverse.rotate(child_pos - base_pos)
        orn = base_orn.inverse * child_orn
        self._constraint = pos, orn

    def get_child(self, base_position):
        base_pos, base_orn = tf(base_position)
        pos, orn = self._constraint
        pos = base_pos + base_orn.rotate(pos)
        orn = base_orn * orn
        return pos, orn


class Rate(object):
    def __init__(self, time_step):
        self._time_step = time_step
        self._next_time = time.time() + time_step

    def sleep(self):
        t = time.time()
        if t < self._next_time:
            time.sleep(self._next_time - t)
        else:
            time.sleep(1e-6)
        self._next_time += self._time_step