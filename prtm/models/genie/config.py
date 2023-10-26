from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    n_timestep: int = 1000
    schedule: str = "cosine"


@dataclass
class ModelConfig:
    c_s: int = 128
    c_p: int = 128
    c_pos_emb: int = 128
    c_timestep_emb: int = 128
    relpos_k: int = 32
    template_type: str = "v1"
    n_pair_transform_layer: int = 5
    include_mul_update: bool = True
    include_tri_att: bool = False
    c_hidden_mul: int = 128
    c_hidden_tri_att: int = 32
    n_head_tri: int = 4
    tri_dropout: float = 0.25
    pair_transition_n: int = 4
    n_structure_layer: int = 5
    n_structure_block: int = 1
    c_hidden_ipa: int = 16
    n_head_ipa: int = 12
    n_qk_point: int = 4
    n_v_point: int = 8
    ipa_dropout: float = 0.1
    n_structure_transition_layer: int = 1
    structure_transition_dropout: float = 0.1


@dataclass
class GenieConfig:
    max_seq_len: int
    model: ModelConfig = ModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()


@dataclass
class Genie128Config(GenieConfig):
    max_seq_len: int = 128


@dataclass
class Genie256Config(GenieConfig):
    max_seq_len: int = 256


@dataclass
class InferenceConfig:
    seq_len: int = 128
    batch_size: int = 1
    verbose: bool = True
