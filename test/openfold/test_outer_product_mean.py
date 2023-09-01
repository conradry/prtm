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

import unittest

import numpy as np
import torch
from proteome.models.folding.openfold.model.outer_product_mean import OuterProductMean
from proteome.models.folding.openfold.utils.tensor_utils import tree_map

from .compare_utils import (
    alphafold_is_installed,
    fetch_alphafold_module_weights,
    get_alphafold_config,
    get_global_pretrained_openfold,
    import_alphafold,
    skip_unless_alphafold_installed,
)
from .config import consts

if alphafold_is_installed():
    alphafold = import_alphafold()
    import haiku as hk
    import jax


def test_shape():
    c = 31

    opm = OuterProductMean(consts.c_m, consts.c_z, c)

    m = torch.rand((consts.batch_size, consts.n_seq, consts.n_res, consts.c_m))
    mask = torch.randint(0, 2, size=(consts.batch_size, consts.n_seq, consts.n_res))
    m = opm(m, mask=mask, chunk_size=None)

    assert m.shape == (consts.batch_size, consts.n_res, consts.n_res, consts.c_z)


@skip_unless_alphafold_installed()
def test_opm_compare():
    def run_opm(msa_act, msa_mask):
        config = get_alphafold_config()
        c_evo = config.model.embeddings_and_evoformer.evoformer
        opm = alphafold.model.modules.OuterProductMean(
            c_evo.outer_product_mean,
            config.model.global_config,
            consts.c_z,
        )
        act = opm(act=msa_act, mask=msa_mask)
        return act

    f = hk.transform(run_opm)

    n_res = consts.n_res
    n_seq = consts.n_seq
    c_m = consts.c_m

    msa_act = np.random.rand(n_seq, n_res, c_m).astype(np.float32) * 100
    msa_mask = np.random.randint(low=0, high=2, size=(n_seq, n_res)).astype(np.float32)

    # Fetch pretrained parameters (but only from one block)]
    params = fetch_alphafold_module_weights(
        "alphafold/alphafold_iteration/evoformer/"
        + "evoformer_iteration/outer_product_mean"
    )
    params = tree_map(lambda n: n[0], params, jax.numpy.DeviceArray)

    out_gt = f.apply(params, None, msa_act, msa_mask).block_until_ready()
    out_gt = torch.as_tensor(np.array(out_gt))

    model = get_global_pretrained_openfold()
    out_repro = (
        model.evoformer.blocks[0]
        .core.outer_product_mean(
            torch.as_tensor(msa_act).cuda(),
            chunk_size=4,
            mask=torch.as_tensor(msa_mask).cuda(),
        )
        .cpu()
    )

    # Even when correct, OPM has large, precision-related errors. It gets
    # a special pass from consts.eps.
    assert torch.max(torch.abs(out_gt - out_repro)) < 5e-4