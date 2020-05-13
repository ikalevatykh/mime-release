import pybullet as pb


class VisualShape(object):
    """
    Visual shape information. You could use this to bridge your own
    rendering method with PyBullet simulation, and synchronize the
    world transforms manually after each simulation step.
    """

    def __init__(self, body_id, link_index, client_id):
        self._body_id = body_id
        self._link_index = link_index
        self._data = None
        self.client_id = client_id

    def _get_data(self, index):
        if self._data is None:
            body_data = pb.getVisualShapeData(self._body_id, physicsClientId=self.client_id)
            link_data = next((i for i in body_data if i[1] == self._link_index))
            self._data = link_data
        return self._data[index]

    def _change(self, **kwargs):
        pb.changeVisualShape(self._body_id, self._link_index,
                             physicsClientId=self.client_id, **kwargs)

    @property
    def rgba_color(self):
        """ URDF color (if any specified) in red/green/blue/alpha. """
        return self._get_data(7)

    @rgba_color.setter
    def rgba_color(self, value):
        self._change(rgbaColor=value)

    @property
    def specular_color(self):
        """ Specular color components, RED, GREEN and BLUE.
        Can be from 0 to large number (>100). """
        return NotImplementedError

    @specular_color.setter
    def specular_color(self, value):
        self._change(specularColor=value)

    @property
    def texture_id(self):
        """ Texture unique id, as returned by 'loadTexture' method.
         This texture will currently only affect the software renderer,
         not the OpenGL visualization window (yet). """
        return NotImplementedError

    @texture_id.setter
    def texture_id(self, value):
        self._change(textureUniqueId=value)

    @property
    def geometry_type(self):
        """ Visual geometry type (TBD). """
        return self._get_data(2)

    @property
    def dimensions(self):
        """ Dimensions (size, local scale) of the geometry. """
        return self._get_data(3)

    @property
    def mesh_file_name(self):
        """ Path to the triangle mesh, if any.
        Typically relative to the URDF file location, but could be absolute. """
        return self._get_data(4)

    @property
    def local_frame_position(self):
        """ Position and orientation of local visual frame,
        relative to link/joint frame. """
        return self._get_data(5), self._get_data(6)


class CollisionShape(object):
    """
    Collision geometry type and other collision shape information
    of existing body base and links.
    """

    def __init__(self, body_id, link_index, client_id):
        self._body_id = body_id
        self._link_index = link_index
        self.client_id = client_id
        self._data = pb.getCollisionShapeData(body_id, link_index,
                                              physicsClientId=client_id)

    @property
    def geometry_type(self):
        """ Visual geometry type (TBD). """
        return self._data[2]

    @property
    def dimensions(self):
        """ Dimensions (size, local scale) of the geometry. """
        return self._data[3]

    @property
    def mesh_file_name(self):
        """ Path to the triangle mesh, if any.
        Typically relative to the URDF file location, but could be absolute. """
        return self._data[4]

    @property
    def local_frame_position(self):
        """ Position and orientation of local visual frame,
        relative to link/joint frame. """
        return self._data[5], self._data[6]

    @property
    def AABB(self):
        return pb.getAABB(self._body_id, physicsClientId=self.client_id)
