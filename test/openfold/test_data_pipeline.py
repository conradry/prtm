# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import shutil

import numpy as np

from proteome.models.openfold.data.data_pipeline import DataPipeline
from proteome.models.openfold.data.templates import TemplateHitFeaturizer

from .compare_utils import (alphafold_is_installed, import_alphafold,
                            skip_unless_alphafold_installed)

if alphafold_is_installed():
    alphafold = import_alphafold()
    import haiku as hk
    import jax


@skip_unless_alphafold_installed()
def test_fasta_compare():
    # AlphaFold runs the alignments and feature processing at the same
    # time, taking forever. As such, we precompute AlphaFold's features
    # using scripts/generate_alphafold_feature_dict.py and the default
    # databases.
    with open(
        os.path.join(
            os.path.dirname(__file__), "test_data/alphafold_feature_dict.pickle"
        ),
        "rb",
    ) as fp:
        alphafold_feature_dict = pickle.load(fp)

    template_featurizer = TemplateHitFeaturizer(
        mmcif_dir=os.path.join(os.path.dirname(__file__), "test_data/mmcifs"),
        max_template_date="2021-12-20",
        max_hits=20,
        kalign_binary_path=shutil.which("kalign"),
        _zero_center_positions=False,
    )

    data_pipeline = DataPipeline(
        template_featurizer=template_featurizer,
    )

    openfold_feature_dict = data_pipeline.process_fasta(
        os.path.join(os.path.dirname(__file__), "test_data/short.fasta"),
        os.path.join(os.path.dirname(__file__), "test_data/alignments"),
    )

    openfold_feature_dict["template_all_atom_masks"] = openfold_feature_dict[
        "template_all_atom_mask"
    ]

    checked = []

    # AlphaFold and OpenFold process their MSAs in slightly different
    # orders, which we compensate for below.
    m_a = alphafold_feature_dict["msa"]
    m_o = openfold_feature_dict["msa"]

    # The first row of both MSAs should be the same, no matter what
    assert np.all(m_a[0, :] == m_o[0, :])

    # Each row of each MSA should appear exactly once somewhere in its
    # counterpart
    matching_rows = np.all((m_a[:, None, ...] == m_o[None, :, ...]), axis=-1)
    assert np.all(np.sum(matching_rows, axis=-1) == 1)

    checked.append("msa")

    # The corresponding rows of the deletion matrix should also be equal
    matching_idx = np.argmax(matching_rows, axis=-1)
    rearranged_o_dmi = openfold_feature_dict["deletion_matrix_int"]
    rearranged_o_dmi = rearranged_o_dmi[matching_idx, :]
    assert np.all(alphafold_feature_dict["deletion_matrix_int"] == rearranged_o_dmi)

    checked.append("deletion_matrix_int")

    # Remaining features have to be precisely equal
    for k, v in alphafold_feature_dict.items():
        assert k in checked or np.all(v == openfold_feature_dict[k])
