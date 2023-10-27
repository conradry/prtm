import os
from pathlib import Path

import pytest
from prtm import protein
from prtm.models.dmpfold2.modeling import DMPFOLD_MODEL_URLS, DMPFoldForFolding
from prtm.query import caching

from ..test_utils import _compare_structures

caching.set_db_path(
    os.path.join(os.path.dirname(__file__), "../cached_queries_for_testing.db")
)


@pytest.mark.parametrize("model_name", list(DMPFOLD_MODEL_URLS.keys()))
def test_dmpfold2_for_folding(model_name):
    # Instantiate OpenFoldForFolding with the given model name
    folder = DMPFoldForFolding(model_name)

    # Generate a random sequence of amino acids
    # MSA queries for this sequence should be in cache or this will be very slow
    # the first time it is run
    sequence = (
        "MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH"
    )

    gt_pdb_file = Path(__file__).parents[0] / f"reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein5.from_pdb_string(gt_pdb_str)

    pred_structure = folder(sequence)[0]
    pred_pdb_str = pred_structure.to_pdb()
    pred_structure = protein.Protein5.from_pdb_string(pred_pdb_str)

    _compare_structures(pred_structure, gt_structure, atol=0.1)
