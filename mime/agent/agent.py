from copy import deepcopy

import numpy as np


class Agent(object):
    def __init__(self, env):
        # prepare default action
        action = env.action_space.sample()
        for k, v in action.items():
            if 'velocity' in k:
                action[k] *= 0
            elif 'torque' in k:
                action[k] *= 0
            elif 'force' in k:
                action[k] *= 0
            elif 'position' in k:
                action[k] = None
            elif 'orientation' in k:
                action[k] = None
            else:
                raise NotImplementedError
        self._action = action
        self._spaces = env.action_space.spaces

    def get_action(self):
        action_update = self.get_action_update()
        if action_update is None:
            return None
        # update action
        for k, v in action_update.items():
            if k in self._action:
                space = self._spaces[k]
                v = np.clip(v, space.low, space.high)
                self._action[k] = v
        return deepcopy(self._action)

    def get_action_update(self):
        raise NotImplementedError
