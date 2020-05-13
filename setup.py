import os
import sys
import numpy
import shutil

from os.path import dirname, realpath, join
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension


if sys.version_info < (3, 5):
    print('Sorry, Python < 3.5 is not supported.')
    sys.exit()


def read_requirements_file(filename):
    req_file_path = join(dirname(realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


def find_datafiles(path):
    return [(join('etc', d), [join(d, f) for f in files])
            for d, folders, files in os.walk(path)]


def extensions():
    ur_kinematics = Extension(
        "ur5_kinematics",
        include_dirs=['mime/plugins/ur_kinematics/include',
                      numpy.get_include()],
        sources=['mime/plugins/ur_kinematics/src/ur_kin.cpp',
                 'mime/plugins/ur_kinematics/src/ur_kin_py.cpp'],
        define_macros=[('UR5_PARAMS', None)],
    )
    return [ur_kinematics]


setup(name='mime',
      version='0.0.1',
      python_requires='>=3.5',
      install_requires=read_requirements_file('requirements.txt'),
      description='MImE - Manipulation Imitation Environments.',
      author='INRIA WILLOW',
      url='https://github.com/ikalevatykh/mime',
      packages=find_packages(),
      ext_modules=extensions(),
      data_files=find_datafiles('mime/assets'),
      )

shutil.copyfile('settings_template.py', 'mime/settings.py')
print('In order to make the repo to work, modify mime/settings.py')
