import numpy as np
import pybullet as pb


class Camera(object):
    def __init__(self, width, height, client_id):
        shape = (height, width)
        self._shape = shape
        self._world_pos = None
        self._proj_mat = None
        self._view_mat = None
        self._attach_link = None
        self._attach_pose = None
        self._render_options = {}
        self._render_flags = 0
        self._rgba = np.zeros(shape + (4, ), dtype=np.uint8)
        self._mask = np.zeros(shape, dtype=np.uint8)
        self._depth = np.zeros(shape, dtype=np.float32)
        self.avg_fps = []
        self.client_id = client_id
        self.infos = {}

    def project(self, fov, near, far):
        """ Apply camera projection matrix.
            Args:
             fov (float): Field of view.
             near float): Near plane distance.
             far (float): Far plane distance.
        """
        h, w = self.shape
        self._near_proj = near
        self._far_proj = far
        self._proj_mat = pb.computeProjectionMatrixFOV(
            fov=fov,
            aspect=w / h,
            nearVal=near,
            farVal=far,
            physicsClientId=self.client_id)

        self.infos.update(dict(
            fov=float(fov),
            aspect = w/h,
            near=near,
            far=far))

    def move_to(self, pos, orn):
        """ Move camera to a specified position in space.
            Args:
             pos (vec3): Camera eye position in Cartesian world coordinates.
             orn (vec4): Camera orientation, quaternion.
        """
        if len(orn) == 3:
            orn = pb.getQuaternionFromEuler(orn)
            mat = pb.getMatrixFromQuaternion(orn)
        else:
            mat = pb.getMatrixFromQuaternion(orn)
        x, y, z = np.array(mat).reshape((3, 3)).T
        self._view_mat = pb.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=pos + z,
            cameraUpVector=y,
            physicsClientId=self.client_id)

    def view_at(self, target, distance, yaw, pitch, roll=0., up='z'):
        """ Move camera to a specified position in space.
            Args:
             target (vec3): Target focus point in Cartesian world coordinates.
             distance (float): Distance from eye to focus point.
             yaw (float): Yaw angle in degrees left / right around up-axis.
             pitch (float): Pitch in degrees up / down.
             roll (float): Roll in degrees around forward vector.
             up (char): Axis up, one of x, y, z.
        """
        self._view_mat = pb.computeViewMatrixFromYawPitchRoll(
            target,
            distance,
            yaw,
            pitch,
            roll,
            'xyz'.index(up),
            physicsClientId=self.client_id)

        self.infos.update(dict(
            target=tuple(target),
            distance=float(distance),
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll)))

    def attach(self, link, pos=(0, 0, 0), orn=(0, 0, 0, 1)):
        """ Attach camera to a link in a specified position.
            Args:
             link (Link): Link to attach.
             pos (vec3): Camera eye position in link coord system.
             orn (vec4): Camera orientation.
        """
        if len(orn) == 3:
            orn = pb.getQuaternionFromEuler(orn)
        self._attach_link = link
        self._attach_pose = pos, orn

    def shot(self):
        """ Computes a RGB image, a depth buffer and a segmentation mask buffer
        with body unique ids of visible objects for each pixel.
        """

        h, w = self._shape
        renderer = pb.ER_BULLET_HARDWARE_OPENGL

        if self._attach_link is not None:
            pos, orn = self._attach_link.state.position
            pos, orn = pb.multiplyTransforms(
                pos, orn, physicsClientId=self.client_id, **self._attach_pose)
            self.move_to(pos, orn)

        w, h, rgba, depth, mask = pb.getCameraImage(
            width=w,
            height=h,
            projectionMatrix=self._proj_mat,
            viewMatrix=self._view_mat,
            renderer=renderer,
            flags=self._render_flags,
            lightDirection=(2, 0, 1),
            lightColor=(1, 1, 1),
            shadow=0,
            physicsClientId=self.client_id,
            **self._render_options)

        if not isinstance(rgba, np.ndarray):
            rgba = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))
            depth = np.array(depth, dtype=np.float32).reshape((h, w))
            mask = np.array(mask, dtype=np.uint8).reshape((h, w))

        self._rgba, self._depth, self._mask = rgba, depth, mask
        # print('cam', sum(self.avg_fps)/len(self.avg_fps))

    @property
    def shape(self):
        """ Width and height tuple. """
        return self._shape

    @property
    def mask(self):
        """ For each pixels the visible object unique id (int).
            (!) Only available when using software renderer. """
        return self._mask

    @property
    def rgba(self):
        """ List of pixel colors in R,G,B,A format, in range char(0..255)
        for each color. """
        return self._rgba

    @property
    def rgb(self):
        """ List of pixel colors in R,G,B format, in range char(0..255)
        for each color. """
        return self._rgba[:, :, :3]

    @property
    def gray(self):
        """ List of pixel grayscales, in range char(0..255). """
        return np.dot(self.rgb, [0.299, 0.587, 0.114]).astype(np.uint8)

    @property
    def depth(self):
        """ Depth buffer, list of floats. """
        near = self._near_proj
        far = self._far_proj  # Camera near, far
        metric_depth = far * near / (far - (far - near) * self._depth)
        return metric_depth

    def depth_uint8(self, kn, kf):
        """ Depth buffer converted to range char(0..255). """
        # kn = 0.35 # Realsense near
        # kf = 1.55 # Realsense far
        # kn = 0.5 # kinect1 near
        # kf = 1.8 # kinect1 far
        # kn is the camera (originally Kinect) near
        # kf is the camera (originally Kinect) far
        metric_depth = self.depth
        metric_depth[metric_depth <= kn] = kf
        kinect_depth = np.clip(metric_depth, kn, kf)
        kinect_depth = (255 * (metric_depth - kn) / (kf - kn)).astype(np.uint8)
        return kinect_depth.astype(np.uint8)

    def mask_link_index(self, flag):
        """ If is enabled, the mask combines the object unique id and link index
        as follows: value = objectUniqueId + (linkIndex+1)<<24.
        """
        if flag:
            self._render_flags |= pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        else:
            self._render_flags &= ~pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

    def casts_shadow(self, flag):
        """ 1 for shadows, 0 for no shadows. """
        self._render_options['shadow'] = 1 if flag else 0

    def set_light_direction(self, vec3):
        """ Light direction. """
        self._render_options['lightDirection'] = vec3

    def set_light_color(self, vec3):
        """ Directional light color in [RED, GREEN, BLUE] in range 0..1. """
        self._render_options['lightColor'] = vec3

    def set_light_distance(self, value):
        """ Distance of the light along the normalized light direction. """
        self._render_options['lightDistance'] = value

    def set_light_ambient_coeff(self, valuem):
        """ Light ambient coefficient. """
        self._render_options['lightAmbientCoeff'] = value

    def set_light_diffuse_coeff(self, value):
        """ Light diffuse coefficient. """
        self._render_options['lightDiffuseCoeff'] = value

    def set_light_specular_coeff(self, value):
        """ Light specular coefficient. """
        self._render_options['lightSpecularCoeff'] = value


class DebugCamera(object):
    @staticmethod
    def view_at(target, distance, yaw, pitch):
        """
            Reset the 3D OpenGL debug visualizer camera.
            Args:
             target (vec3): Target focus point in Cartesian world coordinates.
             distance (float): Distance from eye to focus point.
             yaw (float): Yaw angle in degrees left / right around up-axis.
             pitch (float): Pitch in degrees up / down.
        """
        pb.resetDebugVisualizerCamera(
            cameraTargetPosition=target,
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch)

    @staticmethod
    def get_position():
        """
            Get position of the 3D OpenGL debug visualizer camera.
            Outputs:
             target (vec3): Target focus point in Cartesian world coordinates.
             distance (float): Distance from eye to focus point.
             yaw (float): Yaw angle in degrees left / right around up-axis.
             pitch (float): Pitch in degrees up / down.
        """
        data = pb.getDebugVisualizerCamera()
        yaw = data[8]
        pitch = data[9]
        distance = data[10]
        target = data[11]
        return target, distance, yaw, pitch


class VRCamera(object):
    _pos = (0, 0, 0)
    _orn = (0, 0, 0, 1)

    @staticmethod
    def move_to(pos, orn):
        """
            Move the VR camera to specified position.
            Args:
             pos (vec3): Camera eye position in default coord system.
             orn (vec4): Camera orientation (quaternion or Euler angles).
        """
        if len(orn) == 3:
            orn = pb.getQuaternionFromEuler(orn)
        pb.setVRCameraState(rootPosition=pos, rootOrientation=orn)

    @staticmethod
    def move_step(pos, orn=(0, 0, 0, 1)):
        """
            Move the VR camera by step.
            Args:
             pos (vec3): Linear step.
             orn (vec4): Angular (quaternion or Euler angles).
        """
        if len(orn) == 3:
            orn = pb.getQuaternionFromEuler(orn)
        pos = np.add(VRCamera._pos, pos)
        _, orn = pb.multiplyTransforms((0, 0, 0), VRCamera._orn, (0, 0, 0),
                                       orn)
        VRCamera.move_to(pos, orn)
