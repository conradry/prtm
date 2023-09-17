from pathlib import Path

import pytest

from proteome import protein
from proteome.models.eigenfold.modeling import (
    MODEL_URLS, EigenFoldForFoldSampling)

from ..test_utils import _compare_structures


@pytest.mark.parametrize("model_name", list(MODEL_URLS.keys()))
def test_eigenfold_for_fold_sampling(model_name):
    folder = EigenFoldForFoldSampling(model_name, random_seed=0)

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.ProteinCATrace.from_pdb_string(gt_pdb_str)

    sequence = (
        "MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH"
    )
    pred_structure = folder(sequence)[0]
    pred_pdb = pred_structure.to_pdb()
    pred_structure = protein.ProteinCATrace.from_pdb_string(pred_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.01)
