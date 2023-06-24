from dataclasses import dataclass
from typing import Any, Optional, Union

from proteome.models.folding.esm.data import Alphabet


@dataclass
class StructureModuleConfig:
    c_s: int = 384
    c_z: int = 128
    c_ipa: int = 16
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.1
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5


@dataclass
class FoldingTrunkConfig:
    num_blocks: int = 48
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False

    max_recycles: int = 4
    chunk_size: Optional[int] = None

    structure_module: StructureModuleConfig = StructureModuleConfig()


@dataclass
class ESM2Config:
    num_layers: int = 36
    embed_dim: int = 2560
    attention_heads: int = 20
    alphabet: Alphabet = Alphabet.from_architecture("ESM-1b")
    token_dropout: bool = True


@dataclass
class ESMFoldConfig:
    name: str
    esm_type: str = "esm2_3B"
    fp16_esm: bool = True
    esm_ablate_sequence: bool = False
    esm_input_dropout: float = 0
    embed_aa: bool = True
    bypass_lm: bool = False
    lddt_head_hid_dim: int = 128
    use_esm_attn_map: bool = False

    esm_config: ESM2Config = ESM2Config()
    trunk: Any = FoldingTrunkConfig()
    lddt_head_hid_dim: int = 128


@dataclass
class ESMFoldV0(ESMFoldConfig):
    name: str = "esm_fold_v0"


@dataclass
class ESMFoldV1(ESMFoldConfig):
    name: str = "esm_fold_v1"
