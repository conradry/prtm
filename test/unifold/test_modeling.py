from pathlib import Path

import pytest
from prtm import protein
from prtm.models.unifold.modeling import UniFoldForFolding

from ..test_utils import _compare_structures


MONOMER_MODELS_TO_TEST = ["model_2_ft", "model_1_af2"]
MULTIMER_MODELS_TO_TEST = ["multimer_ft", "multimer_4_af2_v3"]
SYMMETRY_MODELS_TO_TEST = ["uf_symmetry"]


@pytest.mark.parametrize("model_name", MONOMER_MODELS_TO_TEST)
def test_unifold_for_monomer_folding(model_name):
    folder = UniFoldForFolding(model_name, use_templates=True, random_seed=0)

    monomer_sequence = (
        "LILNLRGGAFVSNTQITMADKQKKFINEIQEGDLVRSYSITDETFQQNAVTSIV"
        "KHEADQLCQINFGKQHVVCTVNHRFYDPESKLWKSVCPHPGSGISFLKKYDYLLS"
        "EEGEKLQITEIKTFTTKQPVFIYHIQVENNHNFFANGVLAHAMQVSI"
    )
    monomer_sequence_dict = {"A": monomer_sequence}

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein37.from_pdb_string(gt_pdb_str)

    pred_structure = folder(monomer_sequence_dict)[0]
    pred_pdb_str = pred_structure.to_pdb()
    pred_structure = protein.Protein37.from_pdb_string(pred_pdb_str)

    _compare_structures(pred_structure, gt_structure)


@pytest.mark.parametrize("model_name", MULTIMER_MODELS_TO_TEST)
def test_unifold_for_multimer_folding(model_name):
    folder = UniFoldForFolding(model_name, use_templates=True, random_seed=0)

    complex_sequence_a = (
        "TTPLVHVASVEKGRSYEDFQKVYNAIALKLREDDEYDNYIGYGPVLVRLAWHTSGTW"
        "DKHDNTGGSYGGTYRFKKEFNDPSNAGLQNGFKFLEPIHKEFPWISSGDLFSLGGVTA"
        "VQEMQGPKIPWRCGRVDTPEDTTPDNGRLPDADKDADYVRTFFQRLNMNDREVVALMGAH"
        "ALGKTHLKNSGYEGPWGAANNVFTNEFYLNLLNEDWKLEKNDANNEQWDSKSGYMMLPTDY"
        "SLIQDPKYLSIVKEYANDQDKFFKDFSKAFEKLLENGITFPKDAPSPFIFKTLEEQGL"
    )
    complex_sequence_b = (
        "TEFKAGSAKKGATLFKTRCLQCHTVEKGGPHKVGPNLHGIFGRHSGQAEGYSYTDA"
        "NIKKNVLWDENNMSEYLTNPKKYIPGTKMAIGGLKKEKDRNDLITYLKKACE"
    )
    complex_sequence_dict = {"A": complex_sequence_a, "B": complex_sequence_b}

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein37.from_pdb_string(gt_pdb_str)

    pred_structure = folder(complex_sequence_dict)[0]
    pred_pdb_str = pred_structure.to_pdb()
    pred_structure = protein.Protein37.from_pdb_string(pred_pdb_str)

    _compare_structures(pred_structure, gt_structure)


@pytest.mark.parametrize("model_name", SYMMETRY_MODELS_TO_TEST)
def test_unifold_for_symmetric_folding(model_name):
    folder = UniFoldForFolding(
        model_name, use_templates=True, random_seed=0, symmetry_group="C2"
    )

    symmetric_sequence = (
        "PPYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGD"
        "LTLYQSNTILRHLGRTLGLYGKDQQEAALVDMVNDGVEDLRCKYISLIYTNYEAGKDDYV"
        "KALPGQLKPFETLLSQNQGGKTFIVGDQISFADYNLLDLLLIHEVLAPGCLDAFPLLSAY"
        "VGRLSARPKLKAFLASPEYVNLPINGNGKQ"
    )
    symmetric_sequence_dict = {"A": symmetric_sequence}

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein37.from_pdb_string(gt_pdb_str)

    pred_structure = folder(symmetric_sequence_dict)[0]
    pred_pdb_str = pred_structure.to_pdb()
    pred_structure = protein.Protein37.from_pdb_string(pred_pdb_str)

    _compare_structures(pred_structure, gt_structure)
