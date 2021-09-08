import numpy as np

from ...scene import Body


class TableModder(object):
    def __init__(self, scene, randomize=False, **kwargs):
        self.scene = scene
        self.scene._cage = None
        self.scene._mat = None
        self._cage_urdf = None
        self._randomize = randomize

    def load_cage(self, np_random):
        """
        Set a cage around the robot to match the real setup where walls
        or cage are present. Randomize cage size and pose.
        """
        if self._cage_urdf is not None:
            if self.scene._cage is not None:
                self.scene._cage.remove()

            if self._randomize:
                cage_scaling = np_random.uniform(0.45, 0.7)
                # set cage to random orientation
                cage_pos, cage_ori = np_random.uniform(0, 0.05, 3), (0, 0, 0)
                # cage_ori = np_random.uniform(-np.pi/16, np.pi/16, 3)
                cage_pos[2] += 0.2
            else:
                cage_scaling = 0.55
                cage_pos, cage_ori = (0, 0, 0), (0, 0, 0)

            cage = Body.load(
                self._cage_urdf,
                self.scene.client_id,
                globalScaling=cage_scaling,
                egl=self.scene._load_egl,
            )
            cage.position = cage_pos, cage_ori
            self.scene._cage = cage
        else:
            print("Cage URDF has not been set. Ignoring cage loading.")

    def position_robot_base(self, np_random):
        """
        Randomize the (x, y) position of the robot base.
        """
        xy_noise_base = 0.0  # 0.05 ; use only for regression, maybe not necessary
        robot_base_position = np_random.uniform(
            low=(-xy_noise_base, -xy_noise_base, 0),
            high=(xy_noise_base, xy_noise_base, 0),
            size=(3,),
        )
        self.scene._robot._body.position = robot_base_position, (0, 0, 0, 1)

    def load_mesh(
        self, mesh_path, size_ranges, np_random, mass=0.1, useFixedBase=False
    ):
        """
        Randomize the size of the mesh.
        """
        mesh_size = self.get_size(size_ranges, np_random)

        if mesh_path == "cube":
            mesh = Body.box(
                (mesh_size,) * 3,
                client_id=self.scene.client_id,
                mass=mass,
                collision=True,
                egl=self.scene._load_egl,
            )
        elif mesh_path == "rectangle":
            mesh = Body.box(
                (mesh_size[0], mesh_size[0], mesh_size[1]),
                client_id=self.scene.client_id,
                mass=mass,
                collision=True,
                egl=self.scene._load_egl,
            )
        elif mesh_path == "box":
            mesh = Body.box(
                mesh_size,
                client_id=self.scene.client_id,
                mass=mass,
                collision=True,
                egl=self.scene._load_egl,
            )
        elif mesh_path == "sphere":
            mesh = Body.sphere(
                mesh_size / 2,
                client_id=self.scene.client_id,
                mass=mass,
                collision=True,
                egl=self.scene._load_egl,
            )
        elif mesh_path in ("capsule", "cylinder"):
            assert len(mesh_size) == 2
            mesh = getattr(Body, mesh_path)(
                radius=mesh_size[0],
                height=mesh_size[1],
                client_id=self.scene.client_id,
                mass=mass,
                collision=True,
                egl=self.scene._load_egl,
            )
        else:
            mesh = Body.load(
                mesh_path,
                client_id=self.scene.client_id,
                globalScaling=mesh_size,
                egl=self.scene._load_egl,
                useFixedBase=useFixedBase,
            )

        return mesh, mesh_size

    def get_size(self, size_range, np_random):
        if isinstance(size_range, (float, int)):
            return size_range

        if self._randomize:
            size = np_random.uniform(size_range["low"], size_range["high"])
        else:
            size = (size_range["low"] + size_range["high"]) / 2

        return size
