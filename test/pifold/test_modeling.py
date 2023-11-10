from pathlib import Path

import pytest
from prtm import protein
from prtm.models.pifold import modeling

EXPECTED_SEQUENCES = {
    "base": "EIPIDELKALLFVKALELGDPELMRKVISPDTKMNVNGKEYEGEEIVEFVEEIKKSGTKYKLLSFEKEGDEYLFEVEVKNDGETRDWTVRVEVKDGKIKKVNVTNN",
}


@pytest.mark.parametrize("model_name", list(modeling.PIFOLD_MODEL_CONFIGS.keys()))
def test_pifold_models(model_name: str):
    target_pdb_file = Path(__file__).parents[0] / "5L33.pdb"
    with open(target_pdb_file, "r") as f:
        target_pdb_str = f.read()

    exp_sequence = EXPECTED_SEQUENCES[model_name]
    inverse_folder = modeling.PiFoldForInverseFolding(
        model_name=model_name, random_seed=0
    )
    target_protein = protein.Protein4.from_pdb_string(target_pdb_str)
    sequence = inverse_folder(target_protein)[0]

    assert sequence == exp_sequence
