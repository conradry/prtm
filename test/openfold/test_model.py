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
import unittest

import numpy as np
import torch
import torch.nn as nn

import proteome.models.folding.openfold.utils.feats as feats
from proteome.models.folding.openfold.config import model_config
from proteome.models.folding.openfold.data import data_transforms
from proteome.models.folding.openfold.model.model import AlphaFold
from proteome.models.folding.openfold.utils.tensor_utils import (
    tensor_tree_map,
    tree_map,
)

from .compare_utils import (
    alphafold_is_installed,
    fetch_alphafold_module_weights,
    get_alphafold_config,
    get_global_pretrained_openfold,
    import_alphafold,
    skip_unless_alphafold_installed,
)
from .config import consts
from .data_utils import random_extra_msa_feats, random_template_feats

if alphafold_is_installed():
    alphafold = import_alphafold()
    import haiku as hk
    import jax


def test_dry_run():
    n_seq = consts.n_seq
    n_templ = consts.n_templ
    n_res = consts.n_res
    n_extra_seq = consts.n_extra

    c = model_config("model_1")
    c.model.evoformer_stack.no_blocks = 4  # no need to go overboard here
    c.model.evoformer_stack.blocks_per_ckpt = None  # don't want to set up
    # deepspeed for this test

    model = AlphaFold(c)
    model = model.eval()

    batch = {}
    tf = torch.randint(c.model.input_embedder.tf_dim - 1, size=(n_res,))
    batch["target_feat"] = nn.functional.one_hot(
        tf, c.model.input_embedder.tf_dim
    ).float()
    batch["aatype"] = torch.argmax(batch["target_feat"], dim=-1)
    batch["residue_index"] = torch.arange(n_res)
    batch["msa_feat"] = torch.rand((n_seq, n_res, c.model.input_embedder.msa_dim))
    t_feats = random_template_feats(n_templ, n_res)
    batch.update({k: torch.tensor(v) for k, v in t_feats.items()})
    extra_feats = random_extra_msa_feats(n_extra_seq, n_res)
    batch.update({k: torch.tensor(v) for k, v in extra_feats.items()})
    batch["msa_mask"] = torch.randint(low=0, high=2, size=(n_seq, n_res)).float()
    batch["seq_mask"] = torch.randint(low=0, high=2, size=(n_res,)).float()
    batch.update(data_transforms.make_atom14_masks(batch))
    batch["no_recycling_iters"] = torch.tensor(2.0)

    add_recycling_dims = lambda t: (
        t.unsqueeze(-1).expand(*t.shape, c.data.common.max_recycling_iters)
    )
    batch = tensor_tree_map(add_recycling_dims, batch)

    with torch.no_grad():
        out = model(batch)


@skip_unless_alphafold_installed()
def test_compare():
    def run_alphafold(batch):
        config = get_alphafold_config()
        model = alphafold.model.modules.AlphaFold(config.model)
        return model(
            batch=batch,
            is_training=False,
            return_representations=True,
        )

    f = hk.transform(run_alphafold)

    params = fetch_alphafold_module_weights("")

    with open(
        os.path.join(os.path.dirname(__file__), "test_data/sample_feats.pickle", "rb")
    ) as fp:
        batch = pickle.load(fp)

    out_gt = f.apply(params, jax.random.PRNGKey(42), batch)

    out_gt = out_gt["structure_module"]["final_atom_positions"]
    # atom37_to_atom14 doesn't like batches
    batch["residx_atom14_to_atom37"] = batch["residx_atom14_to_atom37"][0]
    batch["atom14_atom_exists"] = batch["atom14_atom_exists"][0]
    out_gt = alphafold.model.all_atom.atom37_to_atom14(out_gt, batch)
    out_gt = torch.as_tensor(np.array(out_gt.block_until_ready()))

    batch["no_recycling_iters"] = np.array(
        [
            3.0,
            3.0,
            3.0,
            3.0,
        ]
    )
    batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}

    batch["aatype"] = batch["aatype"].long()
    batch["template_aatype"] = batch["template_aatype"].long()
    batch["extra_msa"] = batch["extra_msa"].long()
    batch["residx_atom37_to_atom14"] = batch["residx_atom37_to_atom14"].long()
    batch["template_all_atom_mask"] = batch["template_all_atom_masks"]
    batch.update(data_transforms.atom37_to_torsion_angles("template_")(batch))

    # Move the recycling dimension to the end
    move_dim = lambda t: t.permute(*range(len(t.shape))[1:], 0)
    batch = tensor_tree_map(move_dim, batch)

    with torch.no_grad():
        model = get_global_pretrained_openfold()
        out_repro = model(batch)

    out_repro = tensor_tree_map(lambda t: t.cpu(), out_repro)

    out_repro = out_repro["sm"]["positions"][-1]
    out_repro = out_repro.squeeze(0)

    assert torch.max(torch.abs(out_gt - out_repro)) < 1e-3
