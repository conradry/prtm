# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict
from pathlib import Path

import numpy as np
from proteome import protein
from proteome.models.folding.esm import modeling


def _compare_structures(pred_structure: protein.Protein, gt_structure: protein.Protein):
    pred_structure_dict = asdict(pred_structure)
    gt_structure_dict = asdict(gt_structure)
    for k in pred_structure_dict:
        if isinstance(pred_structure_dict[k], np.ndarray):
            assert np.allclose(pred_structure_dict[k], gt_structure_dict[k], atol=0.1)
        else:
            assert pred_structure_dict[k] == gt_structure_dict[k]


def test_esmfold_models():
    sequence = "MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH"
    model_names = list(modeling.ESMFOLD_MODEL_CONFIGS.keys())
    for model_name in model_names:
        gt_pdb_file = Path(__file__).parents[0] / f"{model_name}_folding.pdb"
        with open(gt_pdb_file, "r") as f:
            gt_pdb_str = f.read()

        print(model_name, str(gt_pdb_file))

        gt_structure = protein.from_pdb_string(gt_pdb_str)

        folder = modeling.ESMForFolding(model_name=model_name, chunk_size=512)
        pred_structure, mean_plddt = folder.fold(sequence)

        # Write to pdb and convert back to ignore atom masking, etc.
        pred_pdb_str = protein.to_pdb(pred_structure)
        pred_structure = protein.from_pdb_string(pred_pdb_str)

        _compare_structures(pred_structure, gt_structure)

""""
def test_esmif_models():
    target_pdb_file = Path(__file__).parents[0] / "esmfold_3B_v1_folding.pdb"
    with open(target_pdb_file, "r") as f:
        target_pdb_str = f.read()

    target_structure =  protein.from_pdb_string(target_pdb_str)
    expected_sequences = {
        "esm_if1_gvp4_t16_142M_UR50": "DQQALIHHHEQEAAQKQALAAKYLDKSKLFSSQGEDTDSAEFAKRAEGESKQAQSHAALAAEGQRLFEQPPPP"
    }

    model_names = list(modeling.ESMIF_MODEL_CONFIGS.keys())
    assert all([m in expected_sequences for m in model_names])

    for model_name in model_names:
        exp_sequence = expected_sequences[model_name]
        inverse_folder = modeling.ESMForInverseFolding(model_name=model_name, random_seed=0)
        sequence, score = inverse_folder.design_sequence(target_structure)
        assert sequence == exp_sequence
"""

