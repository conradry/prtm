from pathlib import Path

import pytest
from prtm import protein
try:
    from prtm.models.protein_seq_des import modeling
    model_configs = list(modeling.PSD_MODEL_CONFIGS.keys())
except ImportError:
    model_configs = []

from ..test_utils import skip_unless_pyrosetta_installed

EXPECTED_SEQUENCES = {
    "conditional_model_0": "SVPETDAVADDADKALVKRDPREVARVITKDTTGADNGNKFQDTLIVDVSRDWHKAGIFMTIESRRKKGDQLEYQVMQSCHGRIDIHTILKKIENGKLHQIYLSGH",
    "conditional_model_1": "SVDETTAVADDADKALVTRDPREVARVISKDATCADNGNKFQDTLIVDVMKDWHKAGIFATIESSRKKGDQLLTQVMQSCHGRVDKVYILKVIENGKLHQIYISKH",
    "conditional_model_2": "SVPETEARADDLDKALVTRDPREVARVISKDATCADNGNKFQDTLIVDVMELWHKAGIFATIESSRKKGDQLETQVMQSCGGRVDIVYILITVANGKIHQIYISKH",
    "conditional_model_3": "SVDETEAVADDADKALVKRDPREVARVTSKDATQAENGNKFQGTLIVDVMNDWHKAGIFATIESRRKKGDQLETQVMQSCGGRHDIVTILIVVENGKLHQIYLSKH",
}


@skip_unless_pyrosetta_installed()
@pytest.mark.parametrize("model_name", model_configs)
def test_protein_seq_des_models(model_name: str):
    target_pdb_file = Path(__file__).parents[0] / "5L33.pdb"
    with open(target_pdb_file, "r") as f:
        target_pdb_str = f.read()

    target_structure = protein.Protein14.from_pdb_string(target_pdb_str)

    exp_sequence = EXPECTED_SEQUENCES[model_name]
    inverse_folder = modeling.ProteinSeqDesForInverseFolding(
        model_name=model_name, random_seed=0
    )
    sequence = inverse_folder(target_structure)[0]
    assert sequence == exp_sequence
