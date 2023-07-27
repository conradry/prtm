from typing import Optional
from dataclasses import dataclass


@dataclass
class ProteinNetConfig:
    num_in_features: int = 21
    num_out_features: int = 20
    hidden_size: int = 128
    adj_input_size: Optional[int] = 2


@dataclass
class InferenceConfig:
    max_sequences: int = 1