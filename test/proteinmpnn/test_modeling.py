from pathlib import Path

import pytest

from proteome import protein
from proteome.models.proteinmpnn import modeling

EXPECTED_SEQUENCES = {
    "vanilla_model-2": "KINKNEKKALEFIKSLENGNPEEMAKVISPNTKLNINGKKYKGKGIIDFIKKIKEKKVKFKLLEYKKEGNKYVFNVEVEYNNKKYLAKVYIKVKNKKIEYVNIEIK",
    "vanilla_model-10": "MVNPDEAIALNFIKSLEEADPELMAKVITPDTELEVNGKKYKGEEIVEFVKEIKKKGVKFKLKSYKKEGDKYVFDIEVSKDGITRNATVSIEVKDGKLDKVVIEDK",
    "vanilla_model-20": "SINPDEKLALDFIKSFEKADPELMAKVITPDTEMEVNGKKFKGKEIVELVKKHKEKGVKFKLKSWKKEGDEYVFEVEVEYKGKKFKATVKILVKDGKIEKIVVELE",
    "vanilla_model-30": "MINEDEKVALDFIKAFEKNDPELMKKVITEETKFEYNGKKFKGEEIVEFVKKLKEKGVKFKLKESKKEGDEYIFKVEVTLNGKTVEATVKIKVKDGKIEEVKVEIK",
    "ca_only_model-2": "KLNKDLKIALEFVKSLEKNDPELMKKVVTPDTEMNINGKKFKGDEIVEFVKKLKEKNIKIKLKSWRKVGNSWEFEIEAEKNGKKKKILVKITVKNGKIEKIEIKYK",
    "ca_only_model-10": "MLDEDEKVALEFIKALEEADPELMAKVISEDTEVEVNGKKFKGKEVVEFVKKLKEKGVKVKFLSSKKEGDKFVFKVEVEYKGKKKIVTVTILVKDGKIEKVKIKYP",
    "ca_only_model-20": "KLNKKEKIALDFIKAIEKLDPELMKKVVGEDTELEVNGKKFKGDEIVEFVKKLKEKGVKVKLKSSKKEGDDFVFDMEVEKNGKKKKVKVTIKVKDGKIEKVKIEIK",
}


@pytest.mark.parametrize("model_name", list(modeling.PROTEINMPNN_MODEL_URLS.keys()))
def test_protein_seq_des_models(model_name: str):
    target_pdb_file = Path(__file__).parents[0] / "5L33.pdb"
    with open(target_pdb_file, "r") as f:
        target_pdb_str = f.read()


    exp_sequence = EXPECTED_SEQUENCES[model_name]
    inverse_folder = modeling.ProteinMPNNForSequenceDesign(
        model_name=model_name, random_seed=0
    )
    target_protein = protein.from_pdb_string(
        target_pdb_str, ca_only=inverse_folder.ca_only, backbone_only=(not inverse_folder.ca_only)
    )
    sequence, score = inverse_folder.design_sequence(target_protein)

    assert sequence == exp_sequence