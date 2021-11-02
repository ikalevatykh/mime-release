import numpy as np

from ...scene import Body


class TableModder(object):
    def __init__(self, scene, randomize=False, **kwargs):
        self.scene = scene
        self.scene._cage = None
        self._cage_urdf = None
        self._randomize = randomize
        self._textures = {}

    def randomize_robot_visual(self, np_random):
        self.scene.robot._body.randomize_shape(
            np_random, self._textures["robot"], per_link=True
        )

    def randomize_cage_visual(self, np_random):
        if self.scene._cage is not None:
            self.scene._cage.randomize_shape(
                np_random, self._textures["background"], per_link=True
            )

    def randomize_table_visual(self, np_random):
        self.scene._table.randomize_shape(
            np_random, self._textures["table"], per_link=True
        )

    def randomize_lighting(self, np_random, camera):
        light_color = np_random.uniform(0, 1, 3)
        light_direction = np.zeros((3))
        while light_direction[0] == 0 and light_direction[1] == 0:
            light_direction[:2] = np_random.uniform(-5, 5, 2)
            light_direction[2] = np_random.uniform(-1, 1, 1)
        light_distance = np_random.uniform(0, 5)
        diffuse_coeff, specular_coeff, ambient_coeff = np_random.uniform(
            0.3, 0.7, size=(3)
        )

        camera.set_lighting(
            light_color,
            light_distance,
            light_direction,
            diffuse_coeff,
            specular_coeff,
            ambient_coeff,
            True,
        )

    def randomize_object_color(self, np_random, obj, obj_color):
        obj.color = obj_color + np.append(np_random.randn(3) * 0.02, 0)

    def load_cage(self, np_random):
        """
        Set a cage around the robot to match the real setup where walls
        or cage are present. Randomize cage size and pose.
        """
        if self._cage_urdf is not None:
            # if self.scene._cage is not None:
            # self.scene._cage.remove()

            # if self._randomize:
            #     cage_scaling = np_random.uniform(0.45, 0.7)
            #     # set cage to random orientation
            #     cage_pos, cage_ori = np_random.uniform(0, 0.05, 3), (0, 0, 0)
            #     # cage_ori = np_random.uniform(-np.pi/16, np.pi/16, 3)
            #     cage_pos[2] += 0.2
            # else:
            #     cage_scaling = 0.55
            #     cage_pos, cage_ori = (0, 0, 0), (0, 0, 0)

            if self.scene._cage is None:
                cage = Body.load(
                    self._cage_urdf,
                    self.scene.client_id,
                    egl=self.scene._load_egl,
                )
                cage.color = (210.0 / 255.0, 213.0 / 255.0, 216.0 / 255.0, 1.0)
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
