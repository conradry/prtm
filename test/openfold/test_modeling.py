import os
from pathlib import Path

import pytest
from prtm import protein
from prtm.models.openfold.modeling import OPENFOLD_MODEL_URLS, OpenFoldForFolding
from prtm.query import caching

from ..test_utils import _compare_structures

caching.set_db_path(
    os.path.join(os.path.dirname(__file__), "../cached_queries_for_testing.db")
)


@pytest.mark.parametrize("model_name", list(OPENFOLD_MODEL_URLS.keys()))
def test_openfold_for_folding(model_name):
    # Instantiate OpenFoldForFolding with the given model name
    openfold = OpenFoldForFolding(model_name, random_seed=0)

    # Generate a random sequence of amino acids
    # MSA queries for this sequence should be in cache or this will be very slow
    # the first time it is run
    sequence = (
        "MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH"
    )

    gt_pdb_file = Path(__file__).parents[0] / f"test_data/reference_{model_name}.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein37.from_pdb_string(gt_pdb_str)

    # Fold the sequence using OpenFoldForFolding
    pred_structure = openfold(sequence)[0]
    pred_pdb_str = pred_structure.to_pdb()
    pred_structure = protein.Protein37.from_pdb_string(pred_pdb_str)

    _compare_structures(pred_structure, gt_structure)
