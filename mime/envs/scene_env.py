import gym
import numpy as np
from gym.utils import seeding


class SceneEnv(gym.Env):
    def __init__(self, scene):
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / scene.dt))
        }
        self._scene = scene
        self._observe = True
        self.seed()

    @property
    def scene(self):
        return self._scene

    @property
    def dt(self):
        return self._scene.dt

    def renders(self, gui=False, shared=False):
        self._scene.renders(gui, shared)

    def observe(self, enable):
        self._observe = enable

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        self._np_random = np_random
        self._seed = seed
        return [seed]

    def reset(self):
        self._scene.reset(self._np_random)
        self._reset(self._scene)
        self._scene.modder_reset(self._np_random)
        if self._observe:
            return self._get_observation(self._scene)
        else:
            return None

    def step(self, action):
        self._set_action(self._scene, action)
        self._scene.step()

        if self._observe:
            obs = self._get_observation(self._scene)
        else:
            obs = None
        success = self._scene.is_task_success()
        rew = self._scene.get_reward(action)
        failure, message = self._scene.is_task_failure()
        done = success or failure
        return obs, rew, done, {'success': success, 'failure_message': message}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        return self._scene.render()

    def close(self):
        self._scene.close()

    def _reset(self, scene):
        pass

    def _get_observation(self, scene):
        raise NotImplementedError

    def _set_action(self, scene, action):
        raise NotImplementedError
