import os

from prtm.models.igfold.utils.general import exists

import antiberty


def get_weights():
    project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
    dest = os.path.join(project_path, "trained_models/AntiBERTy_md_smooth/")

    return dest
