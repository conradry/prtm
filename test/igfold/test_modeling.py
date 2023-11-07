from pathlib import Path

import pytest
from prtm import protein
from prtm.models.igfold import modeling

from ..test_utils import _compare_structures


@pytest.mark.parametrize("model_name", list(modeling.IGFOLD_MODEL_CONFIGS.keys()))
def test_igfold_models(model_name: str):
    heavy_sequence = "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS"
    light_sequence = "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"

    sequences = {}
    if len(heavy_sequence) > 0:
        sequences["H"] = heavy_sequence
    if len(light_sequence) > 0:
        sequences["L"] = light_sequence

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein5.from_pdb_string(gt_pdb_str)

    folder = modeling.IgFoldForFolding(model_name=model_name)
    pred_structure = folder(sequences)[0]

    # Write to pdb and convert back to ignore atom masking, etc.
    pred_pdb_str = pred_structure.to_pdb()
    pred_structure = protein.Protein5.from_pdb_string(pred_pdb_str)

    _compare_structures(pred_structure, gt_structure, atol=0.1)
