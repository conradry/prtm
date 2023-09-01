from pathlib import Path

import pytest

from proteome import protein
from proteome.models.design.protein_seq_des import modeling

from ..test_utils import skip_unless_pyrosetta_installed

EXPECTED_SEQUENCES = {
    "conditional_model_0": "SVPETDAVADDADKALVKRDPREVARVITKDTTGADNGNKFQDTLIVDVSRDWHKAGIFMTIESRRKKGDQLEYQVMQSCHGRIDIHTILKKIENGKLHQIYLSGH",
    "conditional_model_1": "SVDETTAVADDADKALVTRDPREVARVISKDATCADNGNKFQDTLIVDVMKDWHKAGIFATIESSRKKGDQLLTQVMQSCHGRVDKVYILKVIENGKLHQIYISKH",
    "conditional_model_2": "SVPETEARADDLDKALVTRDPREVARVISKDATCADNGNKFQDTLIVDVMELWHKAGIFATIESSRKKGDQLETQVMQSCGGRVDIVYILITVANGKIHQIYISKH",
    "conditional_model_3": "SVDETEAVADDADKALVKRDPREVARVTSKDATQAENGNKFQGTLIVDVMNDWHKAGIFATIESRRKKGDQLETQVMQSCGGRHDIVTILIVVENGKLHQIYLSKH"
}


@skip_unless_pyrosetta_installed()
@pytest.mark.parametrize("model_name", list(modeling.PSD_MODEL_CONFIGS.keys()))
def test_protein_seq_des_models(model_name: str):
    target_pdb_file = Path(__file__).parents[0] / "5L33.pdb"
    with open(target_pdb_file, "r") as f:
        target_pdb_str = f.read()

    target_structure = protein.from_pdb_string(target_pdb_str)

    exp_sequence = EXPECTED_SEQUENCES[model_name]
    inverse_folder = modeling.ProteinSeqDesForSequenceDesign(
        model_name=model_name, random_seed=0
    )
    sequence, score = inverse_folder.design_sequence(target_structure)
    assert sequence == exp_sequence