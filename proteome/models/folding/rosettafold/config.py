from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

@dataclass
class SE3Config:
    """Base configuration for SE3 Transformer"""
    num_layers: int = 2
    num_channels: int = 16
    num_degrees: int = 2
    l0_in_features: int = 32
    l0_out_features: int = 8
    l1_in_features: int = 3
    l1_out_features: int = 3
    num_edge_features: int = 32
    div: int = 2
    n_heads: int = 4


@dataclass
class RefinementConfig(SE3Config):
    """Base configuration for SE3 refinement module"""
    num_layers: int = 3
    num_channels: int = 32
    num_degrees: int = 3
    div: int = 4


@dataclass
class PerformerOptions:
    k_dim: Optional[int] = None
    nb_features: Optional[int] = None
    feature_redraw_interval: int = 1000
    kernel_fn: nn.Module = nn.ReLU(inplace=True)
    qr_uniform_q: bool = False
    no_projection: bool = False


@dataclass
class RoseTTAFoldConfig:
    """Base configuration for RoseTTAFold"""
    n_module: int = 8
    n_module_str: int = 4
    n_module_ref: int = 4
    n_layer: int = 1
    d_msa: int = 384
    d_pair: int = 288
    d_templ: int = 64
    n_head_msa: int = 12
    n_head_pair: int = 8
    n_head_templ: int = 4
    d_hidden: int = 64
    r_ff: int = 4
    n_resblock: int = 1
    p_drop: float = 0.0
    use_templ: bool = True
    performer_N_opts: PerformerOptions = PerformerOptions(nb_features=64)
    performer_L_opts: PerformerOptions = PerformerOptions(nb_features=64) 
    se3_config: SE3Config = SE3Config()
    refinement_config: RefinementConfig = RefinementConfig()
