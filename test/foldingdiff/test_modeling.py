from pathlib import Path

import pytest

from proteome import protein
from proteome.models.design.foldingdiff.modeling import (
    FoldingDiffForStructureDesign, 
    FOLDINGDIFF_MODEL_URLS,
)

from ..test_utils import _compare_structures


@pytest.mark.parametrize("model_name", list(FOLDINGDIFF_MODEL_URLS.keys()))
def test_foldingdiff_for_design(model_name):
    designer = FoldingDiffForStructureDesign(model_name, random_seed=0)

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    generated_structure = designer.design_structure()
    generated_pdb = protein.to_pdb(generated_structure)
    generated_structure = protein.from_pdb_string(generated_pdb)

    _compare_structures(generated_structure, gt_structure, atol=0.01)
