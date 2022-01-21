import os

from pathlib import Path


def assets_path():
    return (Path(__file__) / ".." / "assets").resolve()


def data_path():
    return Path(
        os.path.expandvars("$DATASET")
    ).resolve()  ### ROOT PATH FOR GRASPNET ###
