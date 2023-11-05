from dataclasses import dataclass

@dataclass
class AntiBERTyConfig:
    attention_probs_dropout_prob: float = 0.1
    gradient_checkpointing: bool = False
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 512
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512
    model_type: str = "bert"
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    transformers_version: str = "4.5.1"
    type_vocab_size: int = 2
    use_cache: bool = True
    vocab_size: int = 25
