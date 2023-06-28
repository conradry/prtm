from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from proteome.constants.residue_constants import proteinmppn_restypes


@dataclass
class ProteinMPNNConfig:
    num_letters: int = 21
    node_features: int = 128
    edge_features: int = 128
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    vocab: int = 21
    k_neighbors: int = 64
    augment_eps: float = 0.05
    dropout: float = 0.1
    ca_only: bool = False


@dataclass
class ProteinMPNNCAOnlyConfig(ProteinMPNNConfig):
    ca_only: bool = True


@dataclass
class InferenceConfig:
    # Dictionary with keys that are the index of the target in the batch
    # and the values are a tuple of lists. The first list is the list of
    # masked chains to predict and the second list is the list of visible chains
    # to use as context.
    chain_id_dict: Optional[Dict[int, Tuple[List[int], List[int]]]] = None

    num_seq_per_target: int = 1
    batch_size: int = 1
    sampling_temp: float = 0.1
    omit_aas: str = "X"
    pssm_multi: float = 0.0
    pssm_threshold: float = 0.0
    pssm_log_odds_flag: bool = False
    pssm_bias_flag: bool = False

    # Unbiased
    bias_aas: np.ndarray = np.zeros(len(proteinmppn_restypes), dtype=np.float32)
    omit_aas_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.omit_aas_mask: np.ndarray = np.array(
            [aa in self.omit_aas for aa in proteinmppn_restypes]
        ).astype(np.float32)


@dataclass
class TiedFeaturizeOutput:
    atom_positions: torch.Tensor
    sequence: torch.Tensor
    mask: torch.Tensor
    lengths: np.ndarray
    chain_M: torch.Tensor
    chain_encoding_all: torch.Tensor
    letter_list_list: List[List[str]]
    visible_list_list: List[List[int]]
    masked_list_list: List[List[int]]
    masked_chain_length_list_list: List[List[int]]
    chain_M_pos: torch.Tensor
    omit_AA_mask: torch.Tensor
    residue_idx: torch.Tensor
    dihedral_mask: torch.Tensor
    tied_pos_list_of_lists_list: List[List[List[int]]]
    pssm_coef_all: torch.Tensor
    pssm_bias_all: torch.Tensor
    pssm_log_odds_all: torch.Tensor
    bias_by_res_all: torch.Tensor
    tied_beta: torch.Tensor
