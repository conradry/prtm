"""
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2022 HeliXon Limited
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
# =============================================================================
"""
from dataclasses import dataclass


@dataclass
class PLMConfig:
    alphabet_size: int = 23
    node: int = 1280
    padding_idx: int = 21
    edge: int = 66
    proj_dim: int = 1280 * 2
    attn_dim: int = 256
    num_head: int = 1
    num_relpos: int = 129
    masked_ratio: float = 0.12


@dataclass
class StructureConfig:
    node_dim: int = 384
    edge_dim: int = 128
    num_cycle: int = 8
    num_transition: int = 3
    num_head: int = 12
    num_point_qk: int = 4
    num_point_v: int = 8
    num_scalar_qk: int = 16
    num_scalar_v: int = 16
    num_channel: int = 128
    num_residual_block: int = 2
    hidden_dim: int = 128
    num_bins: int = 50


@dataclass
class BinConfig:
    first_break: float
    last_break: float
    num_bins: int
    ignore_index: int = 0


@dataclass
class ContBinConfig:
    x_min: float
    x_max: float
    x_bins: int


@dataclass
class OmegaFoldModelConfig:
    alphabet_size: int = 21
    node_dim: int = 256
    edge_dim: int = 128
    relpos_len: int = 32
    c: int = 16
    geo_num_blocks: int = 50
    gating: bool = True
    attn_c: int = 32
    attn_n_head: int = 8
    transition_multiplier: int = 4
    activation: str = "ReLU"
    opm_dim: int = 32
    geom_count: int = 2
    geom_c: int = 32
    geom_head: int = 4

    plm: PLMConfig = PLMConfig()
    rough_dist_bin: ContBinConfig = ContBinConfig(x_min=3.25, x_max=20.75, x_bins=16)
    dist_bin: ContBinConfig = ContBinConfig(x_min=2, x_max=65, x_bins=64)
    pos_bin: ContBinConfig = ContBinConfig(x_min=-32, x_max=32, x_bins=64)
    prev_pos: BinConfig = BinConfig(
        first_break=3.25, last_break=20.75, num_bins=16, ignore_index=0
    )
    struct: StructureConfig = StructureConfig()
    struct_embedder: bool = False


@dataclass
class InferenceConfig:
    subbatch_size: int = 512
    num_recycle: int = 10


@dataclass
class OmegaFoldModel1Config(OmegaFoldModelConfig):
    struct_embedder: bool = False


@dataclass
class OmegaFoldModel2Config(OmegaFoldModelConfig):
    struct_embedder: bool = True
