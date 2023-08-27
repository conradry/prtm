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

import numpy as np
import torch
from proteome.models.folding.openfold.model.triangular_multiplicative_update import *
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
    c_z = consts.c_z
    c = 11

    tm = TriangleMultiplicationOutgoing(
        c_z,
        c,
    )

    n_res = consts.c_z
    batch_size = consts.batch_size

    x = torch.rand((batch_size, n_res, n_res, c_z))
    mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))
    shape_before = x.shape
    x = tm(x, mask)
    shape_after = x.shape

    assert shape_before == shape_after


def _tri_mul_compare(incoming=False):
    name = "triangle_multiplication_" + ("incoming" if incoming else "outgoing")

    def run_tri_mul(pair_act, pair_mask):
        config = get_alphafold_config()
        c_e = config.model.embeddings_and_evoformer.evoformer
        tri_mul = alphafold.model.modules.TriangleMultiplication(
            c_e.triangle_multiplication_incoming
            if incoming
            else c_e.triangle_multiplication_outgoing,
            config.model.global_config,
            name=name,
        )
        act = tri_mul(act=pair_act, mask=pair_mask)
        return act

    f = hk.transform(run_tri_mul)

    n_res = consts.n_res

    pair_act = np.random.rand(n_res, n_res, consts.c_z).astype(np.float32)
    pair_mask = np.random.randint(low=0, high=2, size=(n_res, n_res))
    pair_mask = pair_mask.astype(np.float32)

    # Fetch pretrained parameters (but only from one block)]
    params = fetch_alphafold_module_weights(
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/" + name
    )
    params = tree_map(lambda n: n[0], params, jax.numpy.DeviceArray)

    out_gt = f.apply(params, None, pair_act, pair_mask).block_until_ready()
    out_gt = torch.as_tensor(np.array(out_gt))

    model = get_global_pretrained_openfold()
    module = (
        model.evoformer.blocks[0].core.tri_mul_in
        if incoming
        else model.evoformer.blocks[0].core.tri_mul_out
    )
    out_repro = module(
        torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
        mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
        inplace_safe=True,
        _inplace_chunk_size=4,
    ).cpu()

    assert torch.mean(torch.abs(out_gt - out_repro)) < consts.eps


@skip_unless_alphafold_installed()
def test_tri_mul_out_compare():
    _tri_mul_compare()


@skip_unless_alphafold_installed()
def test_tri_mul_in_compare():
    _tri_mul_compare(incoming=True)


def _tri_mul_inplace(incoming=False):
    n_res = consts.n_res

    pair_act = np.random.rand(n_res, n_res, consts.c_z).astype(np.float32)
    pair_mask = np.random.randint(low=0, high=2, size=(n_res, n_res))
    pair_mask = pair_mask.astype(np.float32)

    model = get_global_pretrained_openfold()
    module = (
        model.evoformer.blocks[0].core.tri_mul_in
        if incoming
        else model.evoformer.blocks[0].core.tri_mul_out
    )
    out_stock = module(
        torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
        mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
        inplace_safe=False,
    ).cpu()

    # This has to come second because inference mode is in-place
    out_inplace = module(
        torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
        mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
        inplace_safe=True,
        _inplace_chunk_size=2,
    ).cpu()

    assert torch.mean(torch.abs(out_stock - out_inplace)) < consts.eps


@skip_unless_alphafold_installed()
def test_tri_mul_out_inference():
    _tri_mul_inplace()


@skip_unless_alphafold_installed()
def test_tri_mul_in_inference():
    _tri_mul_inplace(incoming=True)
