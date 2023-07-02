from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from proteome import protein
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
    attention_heads: int = 40
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
    

@dataclass
class GVPConfig:
    top_k_neighbors: int = 30
    node_hidden_dim_scalar: int = 1024
    node_hidden_dim_vector: int = 256
    edge_hidden_dim_scalar: int = 32
    edge_hidden_dim_vector: int = 1
    dropout: float = 0.1
    num_encoder_layers: int = 4


@dataclass
class ESMIFConfig:
    encoder_embed_dim: int = 512
    decoder_embed_dim: int = 512
    dropout: float = 0.1
    alphabet: Alphabet = Alphabet.from_architecture("invariant_gvp")
    encoder_layers: int = 8
    encoder_attention_heads: int = 8
    attention_dropout: float = 0.1
    encoder_ffn_embed_dim: int = 2048
    decoder_embed_dim: int = 512
    decoder_layers: int = 8
    decoder_attention_heads: int = 8
    decoder_ffn_embed_dim: int = 2048
    scale_fc: bool = False
    scale_resids: bool = False
    # Everything in args that startswith "gvp_" will be passed to GVPEncoder
    gvp_config: GVPConfig = GVPConfig()


@dataclass(frozen=True, kw_only=True)
class DesignParams:
    """Design parameters for ProteinMPNN."""
    
    # Binary float mask to indicate designable positions. 1.0 if a position is
    # designable and 0.0 if not.
    confidence: Optional[np.ndarray] = None  # [num_res]

    # A string where each character is a residue type. Masked characters
    # that are not designable are represented by the character "-".
    partial_seq: Optional[str] = None  # [num_res]
    partial_seq_list: Optional[List[str]] = None

    def __post_init__(self):
        if self.confidence is not None and self.partial_seq is not None:
            assert len(self.confidence) == len(self.partial_seq)

        # Fill the "-" characters in the sequence with "<mask>"
        # and assign to partial_seq_list.
        if self.partial_seq is not None:
            self.partial_seq_list = [  #  type: ignore
                "<mask>" if c == "-" else c for c in self.partial_seq
            ]


@dataclass(frozen=True, kw_only=True)
class DesignableProtein(protein.Protein, DesignParams):
    """Protein structure definition with design parameters for ProteinMPNN."""