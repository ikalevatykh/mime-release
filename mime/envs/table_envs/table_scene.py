import math
import numpy as np
from copy import deepcopy

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


from ...scene import Body, DebugCamera, VRCamera, Camera, Scene, UR5, PRLUR5Robot
from .utils import conf_to_radians
from .table_modder import TableModder
from mime.config import assets_path


CAMERA_CFG_PATH = assets_path() / "prl_ur5" / "camera.yml"


class TableScene(Scene):
    """Base scene for tasks with robot on table"""

    def __init__(self, robot_type="UR5", randomize=False, **kwargs):
        super(TableScene, self).__init__(**kwargs)
        arm_dof = {"UR5": 6, "PRL_UR5": 6}[robot_type]

        self._randomize = randomize

        # workspace - box, it depends on robot reachability
        self._workspace = [[0.25, -0.3, 0.02], [0.8, 0.3, 0.3]]

        self._max_tool_velocity = (0.05, 0.25)
        self._max_gripper_velocity = 2.0
        self._max_gripper_force = 5.0
        self._arm_dof = arm_dof

        self._robot_type = robot_type
        self._table = None
        self._cage = None
        self._robot = None
        self._modder = TableModder(self, self._randomize)

        self.cam_params = {
            "target": (0, 0, 0),
            "distance": 1.62,
            "yaw": 90,
            "pitch": -28,
            "fov": 60,
            "near": 0.5,
            "far": 2.0,
        }
        self.cam_rand = {
            "target": ([-0.05] * 3, [0.05] * 3),
            "distance": (-0.05, 0.05),
            "yaw": (-20, 20),
            "pitch": (-6, 6),
            "fov": (-0.0, 0.0),
            "near": (-0.0, 0.0),
            "far": (-0.0, 0.0),
        }

    def _reset(self, np_random):
        """
        Reset the robot position and cage positon.
        """
        robot = self._robot
        cam_params = self.cam_params

        # VRCamera.move_to(pos=(1.0, 0.0, -0.77), orn=(0, 0, np.pi / 2))

        # reset robot state
        self._modder.position_robot_base(np_random)
        if self._robot_type == "UR5":
            robot.arm.kinematics.set_configuration("right up forward")
            robot.gripper.reset("Pinch")

            # set joints initial position
            self._lab_init_qpos = np.array(
                [-2.7569, -1.0896, -1.8057, -1.8186, 1.5689, 3.2652]
            )
            robot.arm.reset(self._lab_init_qpos)
        elif self._robot_type == "PRL_UR5":
            robot.arm.kinematics.set_configuration("right up forward")
            robot.gripper.reset()

            # set joints initial position
            lab_init_qpos = np.array([-42, -94, -114, -99, -118, -147])
            self._lab_init_qpos = np.array([math.radians(v) for v in lab_init_qpos])
            robot.arm.reset(self._lab_init_qpos)

            # right_arm_init_qpos = [-37, -129, -60, 21, -53, -3]
            with open(str(CAMERA_CFG_PATH)) as f:
                self.env_cfg = load(f, Loader=Loader)

            self.camera_cfgs = self.env_cfg["camera_cfgs"]
            self.camera_cfgs = [conf_to_radians(cg) for cg in self.camera_cfgs]
            self.default_right_arm_state = self.camera_cfgs[0]

            # set camera arm to fix pos
            self._right_arm_init_qpos = self.default_right_arm_state.values()

            robot.right_arm.reset(self._right_arm_init_qpos)

    def _load(self):
        """
        Load robot, table and a camera for recording videos
        """

        # add robot
        if self._robot_type == "UR5":
            self._robot = UR5(with_gripper=True, fixed=True, client_id=self.client_id)
            self._workspace = [[0.25, -0.3, 0.02], [0.8, 0.3, 0.3]]
            self._robot.arm.controller.workspace = self._workspace

            self._safe_height = [0.08, 0.15]
            self._gripper_workspace = self._workspace
            table = Body.load("plane.urdf", self.client_id, egl=self._load_egl)
            self._modder._cage_urdf = "ur_description/cage.urdf"

        elif self._robot_type == "PRL_UR5":
            self._robot = PRLUR5Robot(
                with_gripper=True, fixed=True, client_id=self.client_id
            )

            self._safe_height = [0.08, 0.15]

            self._workspace = [[-0.5, -0.05, 0.0], [0.15, 0.22, 0.3]]
            self._robot.arm.controller.workspace = self._workspace

            self._modder._cage_urdf = "prl_ur5/cage.urdf"

            table = Body.load(
                "prl_ur5/table.urdf",
                useFixedBase=True,
                client_id=self.client_id,
            )
            self.cam_params = {
                "pos": [0.0, 0.0, 0.0],
                "orn": [0.0, 0.0, np.pi],
                "fov": 42.5,
                "near": 0.1,
                "far": 10.0,
            }

            self.cam_rand = {
                "pos": ([-0.05] * 3, [0.05] * 3),
                "orn": ([-0.007] * 3, [0.007] * 3),
                "fov": (-0.0, 0.0),
                "near": (-0.0, 0.0),
                "far": (-0.0, 0.0),
            }

        else:
            raise ValueError("Unknown robot type: {}".format(self._robot_type))

        # add table
        self._table = table

    def _step(self, dt):
        self._robot.arm.controller.step(dt)
        self._robot.gripper.controller.step(dt)

    def _render_workspace(self, workspace, color):
        bound = np.array(deepcopy(workspace))
        size = np.ptp(bound, axis=0)
        center = bound.mean(axis=0)
        if np.abs(size[2]) < 1e-3:
            size[2] = 1e-3
        box = Body.box(size=size, client_id=self.client_id, egl=self._load_egl)
        box.color = color
        box.position = center

    @property
    def workspace(self):
        return self._workspace

    @property
    def max_tool_velocity(self):
        return self._max_tool_velocity

    @property
    def max_gripper_velocity(self):
        return self._max_gripper_velocity

    @property
    def max_gripper_force(self):
        return self._max_gripper_force

    @property
    def arm_dof(self):
        return self._arm_dof

    @property
    def robot(self):
        return self._robot

    def render(self):
        self._camera.shot()
        return self._camera.rgba

    def is_task_failure(self):
        # check if joint error (target-real) not too large
        err = self.robot.arm.controller.joints_error
        if not np.allclose(err, 0.0, atol=0.1):
            return True, "Joint error too large."
        return False, ""

    def get_reward(self, action):
        raise NotImplementedError

    def is_task_success(self):
        raise NotImplementedError


def test_scene():
    from itertools import cycle, product
    from time import sleep

    scene = TableScene(robot_type="UR5")
    scene.renders(True)
    np_random = np.random.RandomState(1)
    scene.reset(np_random)
    workspace = np.array(scene.workspace)

    # visualize workspace
    box = Body.box(size=np.ptp(workspace, axis=0), egl=scene._load_egl)
    box.position = np.median(workspace, axis=0)
    box.color = (0, 1, 0, 0.3)

    # move through workspace corners
    pts = cycle(product(range(2), repeat=3))
    for ind in pts:
        pos = workspace[ind, range(3)]
        scene.robot.arm.reset_tool(pos)
        sleep(1)


if __name__ == "__main__":
    test_scene()
