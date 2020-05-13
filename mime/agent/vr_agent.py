import time

import click
import gym
import numpy as np

from .agent import Agent
from .utils import KinematicConstraint, Rate, tf
from ..scene import Body, VR, Marker


class VRAgent(Agent):
    def __init__(self, env, timescale=1):
        super(VRAgent, self).__init__(env)
        scene = env.unwrapped.scene
        self._scene = scene
        self._timescale = int(timescale)
        self._rate = Rate(scene.dt / self._timescale)
        self._controller = VRController(scene)
        self._b_marker = BoundaryMarker(scene)
        self._c_marker = InteractMarker(scene)
        self._counter = 0
        self._action_update = None

    def get_action_update(self):
        self._rate.sleep()
        if self._counter % self._timescale == 0:
            state = self._controller.state()
            if state is None:
                # if controller has not connected to an arm
                self._c_marker.show()
                self._b_marker.hide()
                return None
            self._c_marker.hide()

            dt = self._scene.dt * self._timescale
            low, high = self._scene.workspace
            arm = self._scene.robot.arm
            grip = self._scene.robot.gripper

            arm_target, grip_target = state
            prev_pos, prev_orn = tf(arm.tool_position)
            next_pos, next_orn = tf(arm_target)

            if any(next_pos < low) or any(next_pos > high):
                # if an arm is going outside workspace
                self._b_marker.show()
                return None
            self._b_marker.hide()

            pos_err = next_pos - prev_pos
            orn_err = next_orn * prev_orn.inverse
            grip_err = grip_target - grip.width

            self._action_update = dict(
                linear_velocity=pos_err / dt,
                angular_velocity=np.array(orn_err.axis) * orn_err.angle / dt,
                grip_velocity = 10*(grip_target-0.5)
            )

        self._counter += 1
        return self._action_update


class VRController(object):
    def __init__(self, scene):
        self._arm = scene.robot.arm
        self._grip = scene.robot.gripper
        self._device_id = None
        self._constraint = None

    def state(self):
        for e in VR.events():
            if e.button_was_triggered(VR.GripButton):
                tool_pos, _ = tf(self._arm.tool.state.position)
                ctrl_pos, _ = tf(e.position)
                if np.linalg.norm(ctrl_pos - tool_pos) < 0.05:
                    self._constraint = KinematicConstraint(
                        e.position, self._arm.tool_position)
                    self._device_id = e.controller_id
            elif e.button_was_released(VR.GripButton):
                self._device_id = None

            if self._device_id == e.controller_id:
                arm_target = self._constraint.get_child(e.position)
                grip_target = 10*(1.0 - e.analog)
                return arm_target, grip_target

        return None


class BoundaryMarker(Marker):
    def __init__(self, scene):
        super(BoundaryMarker, self).__init__()
        self.client_id = scene.client_id
        self._center = np.median(scene.workspace, axis=0)
        self._size = np.ptp(scene.workspace, axis=0)

    def make(self):
        boundary = Body.box(size=self._size, client_id=self.client_id)
        boundary.position = self._center
        boundary.color = (1, 0, 0, 0.5)
        return boundary


class InteractMarker(Marker):
    def __init__(self, scene):
        super(InteractMarker, self).__init__()
        self.client_id = scene.client_id
        self._arm = scene.robot.arm

    def make(self):
        pos, _ = self._arm.tool.state.position
        sphere = Body.sphere(radius=0.05, client_id=self.client_id)
        sphere.position = pos
        sphere.color = (0, 0, 0, 0)
        return sphere

    def update(self, marker):
        marker.color = (1.0, 0.5 + 0.5 * np.sin(3 * time.time()), 0.0, 0.2)


@click.command(help='vr_agent env_name [options]')
@click.argument('env_name', type=str)
@click.option('-s', '--seed', default=0, help='seed')
@click.option('-t', '--timescale', type=int, default=10, help='time scale')
def main(env_name, seed, timescale):
    env = gym.make(env_name).unwrapped
    env.observe(False)
    env.renders(shared=True)
    env.seed(seed)
    env.reset()

    agent = VRAgent(env, timescale)
    actions = []
    done = False
    while not done:
        action = agent.get_action()
        if action is not None:
            obs, reward, done, info = env.step(action)
            actions.append(action)

    # check recorded data
    env.seed(seed)
    env.reset()

    for action in actions:
        obs, reward, done, info = env.step(action)

    print(info['success'] and 'Success' or 'Failed')
    env.close()


if __name__ == "__main__":
    main()
