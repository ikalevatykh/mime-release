import tempfile
from PIL import Image

import pybullet as pb

from .utils import augment_path


class Texture(object):
    """
    Helper class for operating on textures.
    """

    @staticmethod
    def load(file_name, client_id):
        """ Load a texture from file and return a non-negative
        texture unique id if the loading succeeds. """
        return pb.loadTexture(augment_path(file_name), physicsClientId=client_id)

    @staticmethod
    def from_array(bitmap, client_id):
        """ Load a texture from memory and return a non-negative
        texture unique id if the loading succeeds. """

        # PyBullet can only load textures from file, so ...
        with tempfile.NamedTemporaryFile() as tmp_file:
            image = Image.fromarray(bitmap, 'RGB')
            image.save(tmp_file, 'PNG')
            tmp_file.flush()
            return pb.loadTexture(tmp_file.name, physicsClientId=client_id)
