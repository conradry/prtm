from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BertForDiffusionConfig:
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: Optional[float] = None
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 384
    initializer_range: float = 0.02
    intermediate_size: int = 768
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 128
    model_type: str = "bert"
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    pad_token_id: int = 0
    position_embedding_type: str = "relative_key"
    transformers_version: str = "4.11.3"
    type_vocab_size: int = 2
    use_cache: bool = False
    vocab_size: int = 30522


@dataclass
class DatasetConfig:
    angles_definitions: str = "canonical-full-angles"
    max_seq_len: int = 128
    timesteps: int = 1000
    variance_schedule: str = "cosine"
    variance_scale: float = 1.0
    mean_offset: np.ndarray = np.array([
        -1.4702034, 0.0361131,  3.1276708,  1.9405054,  2.0354161, 2.1225433
    ])


@dataclass
class InferenceConfig:
    seq_len: int = 128
    dataset_config: DatasetConfig = DatasetConfig()