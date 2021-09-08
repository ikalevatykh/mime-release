import numpy
import math
from numpy.linalg import inv, norm
from pyquaternion import Quaternion
import ur5_kinematics
import pybullet as pb

from .arm import Arm
from .body import Body
from .chain import Chain
from .camera import Camera
from .arm_control import ArmPositionController
from .robotiq_gripper import *


class UR5:
    def __init__(
        self, client_id, with_gripper=False, pos=(0, 0, 0), orn=(0, 0, 0, 1), fixed=True
    ):

        model = "willbot_paris.urdf"

        flags = pb.URDF_USE_INERTIA_FROM_FILE
        flags |= pb.URDF_USE_SELF_COLLISION
        flags |= pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        flags |= pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        # flags |= pb.URDF_ENABLE_SLEEPING
        body = Body.load(
            "ur_description/" + model,
            flags=flags,
            useFixedBase=fixed,
            client_id=client_id,
        )
        body.position = pos, orn

        arm = Arm(body, tip_link_name="tool")
        arm.controller = ArmPositionController(arm, gains=0.1)
        arm._kinematics = UR5Kinematics(arm._chain)

        gripper = None
        if with_gripper:
            gripper = RobotiqGripper(body)

        self._body = body
        self._arm = arm
        self._gripper = gripper
        self._wrist_camera = None
        self.client_id = client_id

    def enable_wrist_camera(self, width=320, height=240):
        link = self._body.link("wrist_camera")
        cam = Camera(width, height)
        cam.attach(link=link, orn=(0, 0, np.pi))
        self._wrist_camera = cam

    @property
    def arm(self):
        return self._arm

    @property
    def gripper(self):
        return self._gripper

    @property
    def wrist_camera(self):
        return self._wrist_camera


class UR5Kinematics:
    """
    These kinematics find the tranfrom from the base link to the end effector.
    """

    def __init__(self, chain: Chain, prefix=""):
        body = Body(chain.body_id, chain.client_id)

        def H(link_name):
            state = body.link(link_name).state
            return _homogenous(*state.world_link_frame_position)

        Tw0 = H(f"{prefix}base_link")

        T0w = np.linalg.inv(Tw0)

        Tw6 = H(f"{prefix}ee_link")
        Twt = H(f"{prefix}gripper_grasp_frame")
        Tt6 = np.dot(np.linalg.inv(Twt), Tw6)
        T6t = np.linalg.inv(Tt6)

        self.Tw0 = Tw0
        self.T0w = T0w
        self.Tt6 = Tt6
        self.T6t = T6t
        self.lower, self.upper = chain.limits
        self.kin_indicies = np.array([0, 0, 0])

    def forward(self, q):
        """Find the tranfrom from the base link to the tool link.

        Arguments:
            q {list(6)} -- joint angles.

        Returns:
            (list(3), list(4)) -- position and orientation of the tool link.
        """
        q = np.float64(q)
        T06 = np.zeros((4, 4), dtype=np.float64)
        ur5_kinematics.forward(q, T06)
        T0t = np.dot(T06, self.T6t)
        Twt = np.dot(self.Tw0, T0t)
        return _pos_orn(Twt)

    def set_configuration(self, desc):
        """Specify target kinematic configuration.

        Arguments:
            desc {str} -- configuraton description like 'right up forward'.
                            Posiible options: left/right shoulder,
                            up/down elbow, forward/backward wrist
        """

        config = dict(
            right=(0, 1),
            left=(0, -1),
            up=(1, 1),
            down=(1, -1),
            forward=(2, 1),
            backward=(2, -1),
        )

        indicies = np.array([0, 0, 0])
        for name in desc.split(" "):
            assert name in config, "Unknown kinematic index: {}".format(name)
            i, val = config[name]
            indicies[i] = val

        self.kin_indicies = indicies

    def _get_configuration(self, q):
        right = q[0] < -np.pi / 2
        up = -np.pi <= q[1] < 0
        forward = -np.pi <= q[3] < 0.1
        return [0, 1 if up else -1, 1 if forward else -1]

    def inverse_all(self, pos, orn, q_init):
        """Find all posiible solutions for inverse kinematics.

        Arguments:
            pos {list(3)} -- target tool position.
            orn {list(4)} -- target tool orientation.

        Keyword Arguments:
            q6_des {float} -- An optional parameter which designates what the q6 value should take
                in case of an infinite solution on that joint. (default: {0.0})

        Returns:
            list(N,6) -- solutions.
        """

        Twt = _homogenous(pos, orn)
        T0t = np.dot(self.T0w, Twt)
        T06 = np.float64(np.dot(T0t, self.Tt6))

        q6_des = q_init[5] if q_init is not None else 0.0
        q_sol = np.zeros((8, 6), dtype=np.float64)
        n = ur5_kinematics.inverse(T06, q_sol, q6_des)
        if n == 0:
            return []

        q_sol = q_sol[:n]

        q_sol[q_sol > self.upper] -= 2 * np.pi
        q_sol[q_sol < self.lower] += 2 * np.pi

        mask = np.any((self.lower <= q_sol) & (q_sol <= self.upper), axis=1)

        mask &= q_sol[:, 1] < 0
        q_sol = q_sol[mask]

        mask = [all(self._get_configuration(q) * self.kin_indicies >= 0) for q in q_sol]

        q_sol = q_sol[mask]

        if np.any(q_sol) and q_init is not None:
            weights = [1, 1, 1, 2, 1, 0.5]
            dist = norm((q_sol - q_init) * weights, axis=1)
            q_sol = q_sol[dist.argsort()]

        return q_sol

    def inverse(self, pos, orn, q_init=None):
        """Find inverse kin solution nearest to q_init.

        Arguments:
            pos {list(3)} -- target tool position.
            orn {list(4)} -- target tool orientation.

        Keyword Arguments:
            q_init {list(6)} -- initial solution (default: zeros(6))

        Returns:
            list(6) / None -- joint positions or None if solution not found
        """

        if q_init is None:
            q_init = np.zeros(6, dtype=np.float64)

        q_sol = self.inverse_all(pos, orn, q_init)

        if np.any(q_sol):
            return q_sol[0]

    def forward_vel(self, q, dq, dt):
        pos0, orn0 = self.forward(q)
        pos1, orn1 = self.forward(q + np.array(dq) * dt)

        diff = _quat(orn1) * _quat(orn0).inverse
        axis, angle = diff.get_axis(undefined=[0, 0, 0]), diff.angle

        v = np.subtract(pos1, pos0) / dt
        w = np.array(axis) * angle / dt
        return v, w

    def inverse_vel(self, q, v, w, dt):
        pos0, orn0 = self.forward(q)
        pos1 = pos0 + v * dt
        orn1 = _orn(_quat([*(0.5 * w * dt), 1.0]) * _quat(orn0))

        q1 = self.inverse(pos1, orn1, q)
        if q1 is not None:
            dq = np.subtract(q1, q) / dt
            return dq
        return np.zeros(6)


def _quat(orn):
    return Quaternion(orn[3], *orn[:3])


def _orn(q):
    return [q[1], q[2], q[3], q[0]]


def _homogenous(pos, orn):
    mat = _quat(orn).transformation_matrix
    mat[:3, 3] = pos
    return mat


def _pos_orn(mat):
    pos = mat[:3, 3]
    q = Quaternion(matrix=mat)
    return pos, _orn(q)
