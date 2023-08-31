from dataclasses import asdict

import pytest
import numpy as np

from proteome import protein


def _compare_structures(
    pred_structure: protein.Protein,
    gt_structure: protein.Protein,
    atol: float = 0.1,
):
    pred_structure_dict = asdict(pred_structure)
    gt_structure_dict = asdict(gt_structure)
    for k in pred_structure_dict:
        if isinstance(pred_structure_dict[k], np.ndarray):
            assert np.allclose(pred_structure_dict[k], gt_structure_dict[k], atol=atol)
        else:
            assert pred_structure_dict[k] == gt_structure_dict[k]


def pyrosetta_is_installed():
    try:
        import pyrosetta  # noqa: F401
    except ImportError:
        return False

    return True


def skip_unless_pyrosetta_installed():
    return pytest.mark.skipif(not pyrosetta_is_installed(), reason="Requires PyRosetta")
