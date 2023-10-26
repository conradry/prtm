from dataclasses import dataclass


@dataclass
class R3:
    min_b: float = 0.1
    max_b: float = 20.0
    coordinate_scaling: float = 0.1


@dataclass
class SO3:
    num_omega: int = 1000
    num_sigma: int = 1000
    min_sigma: float = 0.1
    max_sigma: float = 1.5
    schedule: str = "logarithmic"
    cache_dir: str = ".cache/"
    use_cached_score: bool = False


@dataclass
class Embed:
    index_embed_size: int = 32
    aatype_embed_size: int = 64
    embed_self_conditioning: bool = True
    num_bins: int = 22
    min_bin: float = 1e-05
    max_bin: float = 20.0


@dataclass
class Ipa:
    c_s: str = 256
    c_z: str = 128
    c_hidden: int = 256
    c_skip: int = 64
    no_heads: int = 8
    no_qk_points: int = 8
    no_v_points: int = 12
    seq_tfmr_num_heads: int = 4
    seq_tfmr_num_layers: int = 2
    num_blocks: int = 4
    coordinate_scaling: float = 0.1


@dataclass
class ScoreNetworkConfig:
    node_embed_size: int = 256
    edge_embed_size: int = 128
    dropout: float = 0.0
    embed: Embed = Embed()
    ipa: Ipa = Ipa()


@dataclass
class DiffuserConfig:
    diffuse_trans: bool = True
    diffuse_rot: bool = True
    r3: R3 = R3()
    so3: SO3 = SO3()


@dataclass
class DiffusionParams:
    num_t: int = 500
    noise_scale: float = 0.1
    min_t: float = 0.01


@dataclass
class InferenceConfig:
    length: int = 64
    diffusion_params: DiffusionParams = DiffusionParams()
