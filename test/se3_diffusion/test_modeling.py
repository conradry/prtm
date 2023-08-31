from pathlib import Path

import pytest

from proteome import protein
from proteome.models.design.se3_diffusion import modeling, config

from ..test_utils import _compare_structures


@pytest.mark.parametrize("model_name", list(modeling.SE3_MODEL_CONFIGS.keys()))
def test_se3_diffusion_models(model_name: str):
    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    structure_designer = modeling.SE3DiffusionForStructureDesign(
        model_name=model_name, random_seed=0
    )
    pred_structure = structure_designer.design_structure(
        config.InferenceConfig(length=40)
    )

    # Write to pdb and convert back to ignore atom masking, etc.
    pred_pdb_str = protein.to_pdb(pred_structure)
    pred_structure = protein.from_pdb_string(pred_pdb_str)

    _compare_structures(pred_structure, gt_structure)