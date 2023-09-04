from pathlib import Path

import pytest

from proteome import protein
from proteome.models.genie import config
from proteome.models.genie.modeling import (GENIE_MODEL_URLS,
                                            GenieForStructureDesign)

from ..test_utils import _compare_structures


@pytest.mark.parametrize("model_name", list(GENIE_MODEL_URLS.keys()))
def test_genie_for_design(model_name):
    designer = GenieForStructureDesign(model_name, random_seed=0)

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    generated_structure = designer.design_structure(config.InferenceConfig(seq_len=100))
    generated_pdb = protein.to_pdb(generated_structure)
    generated_structure = protein.from_pdb_string(generated_pdb)

    _compare_structures(generated_structure, gt_structure, atol=0.01)
