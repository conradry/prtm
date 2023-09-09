import pytest
import numpy as np

from proteome import protein


def _compare_structures(
    pred_structure: protein._Protein,
    gt_structure: protein._Protein,
    atol: float = 0.1,
):
    for field in pred_structure.fields:
        if isinstance(getattr(pred_structure, field), np.ndarray):
            assert np.allclose(
                getattr(pred_structure, field), getattr(gt_structure, field), atol=atol
            )
        else:
            assert getattr(pred_structure, field) == getattr(gt_structure, field)


def pyrosetta_is_installed():
    try:
        import pyrosetta  # noqa: F401
    except ImportError:
        return False

    return True


def skip_unless_pyrosetta_installed():
    return pytest.mark.skipif(not pyrosetta_is_installed(), reason="Requires PyRosetta")
