# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Contents of this file were adapted from the open source fairseq repository.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict

import argparse
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from proteome.models.folding.esm.modules import SinusoidalPositionalEmbedding
from proteome.models.folding.esm.inverse_folding.features import GVPInputFeaturizer, DihedralFeatures
from proteome.models.folding.esm.inverse_folding.gvp_encoder import GVPEncoder
from proteome.models.folding.esm.inverse_folding.transformer_layer import TransformerEncoderLayer
from proteome.models.folding.esm.inverse_folding.util import nan_to_num, get_rotation_frames, rotate, rbf
from proteome.models.folding.esm import config


class GVPTransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        alphabet (~fairseq.data.Dictionary): encoding alphabet
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg: config.ESMIFConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.alphabet = cfg.alphabet

        self.dropout_module = nn.Dropout(cfg.dropout)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim,
            self.padding_idx,
        )
        self.embed_gvp_input_features = nn.Linear(15, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim)

        self.gvp_encoder = GVPEncoder(cfg.gvp_config)
        gvp_out_dim = cfg.gvp_config.node_hidden_dim_scalar + (3 *
                cfg.gvp_config.node_hidden_dim_vector)
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def build_encoder_layer(self, cfg: config.ESMIFConfig):
        return TransformerEncoderLayer(cfg)

    def forward_embedding(self, coords, padding_mask, confidence):
        """
        Args:
            coords: N, CA, C backbone coordinates in shape length x 3 (atoms) x 3 
            padding_mask: boolean Tensor (true for padding) of shape length
            confidence: confidence scores between 0 and 1 of shape length
        """
        components = dict()
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coords = nan_to_num(coords)
        mask_tokens = (
            padding_mask * self.alphabet.padding_idx + 
            ~padding_mask * self.alphabet.get_idx("<mask>")
        )
        components["tokens"] = self.embed_tokens(mask_tokens) * self.embed_scale
        components["diherals"] = self.embed_dihedrals(coords)

        # GVP encoder
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, padding_mask, confidence)
        R = get_rotation_frames(coords)
        # Rotate to local rotation frame for rotation-invariance
        gvp_out_features = torch.cat([
            gvp_out_scalars,
            rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)

        components["confidence"] = self.embed_confidence(
             rbf(confidence, 0., 1.))

        # In addition to GVP encoder outputs, also directly embed GVP input node
        # features to the Transformer
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False)
        features = torch.cat([
            scalar_features,
            rotate(vector_features, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_input_features"] = self.embed_gvp_input_features(features)

        embed = sum(components.values())
        # for k, v in components.items():
        #     print(k, torch.mean(v, dim=(0,1)), torch.std(v, dim=(0,1)))

        x = embed
        x = x + self.embed_positions(mask_tokens)
        x = self.dropout_module(x)
        return x, components 

    def forward(
        self,
        coords,
        encoder_padding_mask,
        confidence,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            coords (Tensor): backbone coordinates
                shape batch_size x num_residues x num_atoms (3 for N, CA, C) x 3
            encoder_padding_mask (ByteTensor): the positions of
                  padding elements of shape `(batch_size x num_residues)`
            confidence (Tensor): the confidence score of shape (batch_size x
                num_residues). The value is between 0. and 1. for each residue
                coordinate, or -1. if no coordinate is given
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(num_residues, batch_size, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch_size, num_residues)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch_size, num_residues, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(num_residues, batch_size, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(coords,
                encoder_padding_mask, confidence)
        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # alphabet
            "encoder_states": encoder_states,  # List[T x B x C]
        }
