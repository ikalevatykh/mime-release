import sys

from pathlib import Path
from pybullet_data import getDataPath


def augment_path(file_name):
    roots = [
        Path(__file__).parent.parent / 'assets',  # local
        Path(sys.prefix) / 'etc' / 'mime' / 'assets',  # global
        Path(getDataPath())]
    for root in roots:
        path = root / file_name
        if path.exists():
            return path
    return Path(file_name)
