from pathlib import Path

import pytest
from prtm import protein
from prtm.models.proteinsolver import modeling

EXPECTED_SEQUENCES = {
    "model_0": "MLPAVEAAARAFLEALESGSPELLRELLEPEVTIQAEGFELTGEEVVAFVEEVTELGTRWRLTSFERVGGTWTFALRVTVDGETLTFRVTLDVREGRISRLQLTLR",
}


@pytest.mark.parametrize("model_name", list(modeling.PS_MODEL_URLS.keys()))
def test_protein_seq_des_models(model_name: str):
    target_pdb_file = Path(__file__).parents[0] / "5L33.pdb"
    with open(target_pdb_file, "r") as f:
        target_pdb_str = f.read()

    target_structure = protein.Protein27.from_pdb_string(target_pdb_str)

    exp_sequence = EXPECTED_SEQUENCES[model_name]
    inverse_folder = modeling.ProteinSolverForInverseFolding(
        model_name=model_name, random_seed=0
    )
    sequence = inverse_folder(target_structure)[0]
    assert sequence == exp_sequence
