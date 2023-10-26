# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from prtm.models.esm import config
from prtm.models.esm.inverse_folding.features import GVPGraphEmbedding
from prtm.models.esm.inverse_folding.gvp_modules import GVPConvLayer, LayerNorm
from prtm.models.esm.inverse_folding.gvp_utils import unflatten_graph


class GVPEncoder(nn.Module):
    def __init__(self, cfg: config.GVPConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_graph = GVPGraphEmbedding(cfg)

        node_hidden_dim = (cfg.node_hidden_dim_scalar, cfg.node_hidden_dim_vector)
        edge_hidden_dim = (cfg.edge_hidden_dim_scalar, cfg.edge_hidden_dim_vector)

        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(
                node_hidden_dim,
                edge_hidden_dim,
                drop_rate=cfg.dropout,
                vector_gate=True,
                attention_heads=0,
                n_message=3,
                conv_activations=conv_activations,
                n_edge_gvps=0,
                eps=1e-4,
                layernorm=True,
            )
            for i in range(cfg.num_encoder_layers)
        )

    def forward(self, coords, coord_mask, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
            coords, coord_mask, padding_mask, confidence
        )

        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(
                node_embeddings, edge_index, edge_embeddings
            )

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings
