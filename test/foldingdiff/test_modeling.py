from pathlib import Path

import pytest
from prtm import protein
from prtm.models.foldingdiff.modeling import (
    FOLDINGDIFF_MODEL_URLS,
    FoldingDiffForStructureDesign,
)

from ..test_utils import _compare_structures


@pytest.mark.parametrize("model_name", list(FOLDINGDIFF_MODEL_URLS.keys()))
def test_foldingdiff_for_design(model_name):
    designer = FoldingDiffForStructureDesign(model_name, random_seed=0)

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein3.from_pdb_string(gt_pdb_str)

    generated_structure = designer()[0]
    generated_pdb = generated_structure.to_pdb()
    generated_structure = protein.Protein3.from_pdb_string(generated_pdb)

    _compare_structures(generated_structure, gt_structure, atol=0.01)
