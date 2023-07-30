from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from transformers import BertConfig

DEFAULT_BERT_CONFIG = BertConfig(
    attention_probs_dropout_prob=0.1,
    classifier_dropout=None,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=384,
    initializer_range=0.02,
    intermediate_size=768,
    layer_norm_eps=1e-12,
    max_position_embeddings=128,
    model_type="bert",
    num_attention_heads=12,
    num_hidden_layers=12,
    pad_token_id=0,
    position_embedding_type="relative_key",
    transformers_version="4.11.3",
    type_vocab_size=2,
    use_cache=False,
    vocab_size=30522,
)


@dataclass
class FoldingDiffConfig:
    ft_is_angular: List[bool]
    time_encoding: Literal["gaussian_fourier", "sinusoidal"]
    decoder: Literal["mlp", "linear"]
    ft_names: Optional[List[str]] = None
    bert_config: BertConfig = DEFAULT_BERT_CONFIG


FoldingDiffCathConfig = FoldingDiffConfig(
    ft_is_angular=[True, True, True, True, True, True],
    ft_names=[
        "phi",
        "psi",
        "omega",
        "tau",
        "CA:C:1N",
        "C:1N:1CA",
    ],
    time_encoding="gaussian_fourier",
    decoder="mlp",
)


@dataclass
class DatasetConfig:
    angles_definitions: str = "canonical-full-angles"
    max_seq_len: int = 128
    timesteps: int = 1000
    variance_schedule: str = "cosine"
    variance_scale: float = 1.0
    mean_offset: np.ndarray = np.array(
        [-1.4702034, 0.0361131, 3.1276708, 1.9405054, 2.0354161, 2.1225433]
    )


@dataclass
class InferenceConfig:
    seq_len: int = 128
    dataset_config: DatasetConfig = DatasetConfig()
