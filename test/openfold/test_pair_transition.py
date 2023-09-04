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

from proteome.models.openfold.model.pair_transition import PairTransition
from proteome.models.openfold.utils.tensor_utils import tree_map

from .compare_utils import (alphafold_is_installed,
                            fetch_alphafold_module_weights, import_alphafold,
                            skip_unless_alphafold_installed)
from .config import consts

if alphafold_is_installed():
    alphafold = import_alphafold()
    import haiku as hk
    import jax


def test_shape():
    c_z = consts.c_z
    n = 4

    pt = PairTransition(c_z, n)

    batch_size = consts.batch_size
    n_res = consts.n_res

    z = torch.rand((batch_size, n_res, n_res, c_z))
    mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))
    shape_before = z.shape
    z = pt(z, mask=mask, chunk_size=None)
    shape_after = z.shape

    assert shape_before == shape_after


@skip_unless_alphafold_installed()
def test_compare():
    def run_pair_transition(pair_act, pair_mask):
        config = get_alphafold_config()
        c_e = config.model.embeddings_and_evoformer.evoformer
        pt = alphafold.model.modules.Transition(
            c_e.pair_transition,
            config.model.global_config,
            name="pair_transition",
        )
        act = pt(act=pair_act, mask=pair_mask)
        return act

    f = hk.transform(run_pair_transition)

    n_res = consts.n_res

    pair_act = np.random.rand(n_res, n_res, consts.c_z).astype(np.float32)
    pair_mask = np.ones((n_res, n_res)).astype(np.float32)  # no mask

    # Fetch pretrained parameters (but only from one block)]
    params = fetch_alphafold_module_weights(
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"
        + "pair_transition"
    )
    params = tree_map(lambda n: n[0], params, jax.numpy.DeviceArray)

    out_gt = f.apply(params, None, pair_act, pair_mask).block_until_ready()
    out_gt = torch.as_tensor(np.array(out_gt.block_until_ready()))

    model = get_global_pretrained_openfold()
    out_repro = (
        model.evoformer.blocks[0]
        .core.pair_transition(
            torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
            chunk_size=4,
            mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
        )
        .cpu()
    )

    assert torch.max(torch.abs(out_gt - out_repro) < consts.eps)
