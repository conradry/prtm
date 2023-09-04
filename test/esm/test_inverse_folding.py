# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import numpy as np
import torch
from proteome.models.esm.inverse_folding import util
from proteome.models.esm.modeling import ESMForInverseFolding
from scipy.stats import special_ortho_group
from tqdm import tqdm


def test_esm_if1():
    example_file = (
        Path(__file__).absolute().parent / "inverse_folding_test_example.json"
    )
    with open(example_file) as f:
        examples = json.load(f)

    inverse_folder = ESMForInverseFolding()
    model = inverse_folder.model.cpu()
    alphabet = inverse_folder.cfg.alphabet
    batch_converter = util.CoordBatchConverter(alphabet)

    with torch.no_grad():
        print("Testing batch inference on 3 examples...")
        # Test batch with multiple examples
        batch = [(e["coords"], None, e["seq"]) for e in examples[:3]]
        coords, confidence, strs, tokens, padding_mask = batch_converter(batch)
        prev_output_tokens = tokens[:, :-1]
        target = tokens[:, 1:]
        logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coord_mask = coord_mask[:, 1:-1]
        avgloss = torch.sum(loss * coord_mask) / torch.sum(coord_mask)
        expected_ppl = 4.40
        np.testing.assert_allclose(
            expected_ppl,
            torch.exp(avgloss).item(),
            atol=1e-02,
        )

        print("Testing on 10 examples from validation set...")
        # Test batch with single example
        for example in tqdm(examples):
            batch = [(example["coords"], None, example["seq"])]
            coords, confidence, strs, tokens, padding_mask = batch_converter(batch)
            prev_output_tokens = tokens[:, :-1]
            target = tokens[:, 1:]
            logits, _ = model.forward(
                coords, padding_mask, confidence, prev_output_tokens
            )
            assert torch.any(torch.isnan(logits)) == False

            # Test equivariance
            R = special_ortho_group.rvs(3)
            R = torch.tensor(R, dtype=torch.float32)
            coords = torch.matmul(coords, R)
            logits_rotated, _ = model.forward(
                coords, padding_mask, confidence, prev_output_tokens
            )
            np.testing.assert_allclose(
                logits.detach().numpy(), logits_rotated.detach().numpy(), atol=1e-01
            )
