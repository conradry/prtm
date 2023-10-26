from dataclasses import dataclass


@dataclass
class InferenceConfig:
    elbo: bool = True
    alpha: float = 1
    beta: float = 3
    step: float = 0.5
    elbo_step: float = 0.2
    Hf: float = 2
    tmin: float = 0.01
    cutoff: float = 5
    kmin: int = 5
    schedule_type: str = "rate"

    def __post_init__(self):
        assert self.schedule_type in ["entropy", "rate"]


@dataclass
class ScoreModelConfig:
    sde_a: float = 3 / (3.8**2)
    sde_b: float = 0
    resi_conv_layers: int = 6
    resi_ns: int = 32
    resi_nv: int = 4
    resi_ntps: int = 16
    resi_ntpv: int = 4
    resi_fc_dim: int = 128
    resi_pos_emb_dim: int = 16
    lin_nf: int = 1
    lin_self: bool = False
    attention: bool = False
    sh_lmax: int = 2
    order: int = 1
    t_emb_dim: int = 32
    t_emb_type: str = "sinusoidal"
    radius_emb_type: str = "gaussian"
    radius_emb_dim: int = 50
    radius_emb_max: float = 50
    tmin: float = 0.001
    tmax: float = 1e6
    no_radius_sqrt: bool = False
    parity: bool = True
    lm_edge_dim: int = 128
    lm_node_dim: int = 256
