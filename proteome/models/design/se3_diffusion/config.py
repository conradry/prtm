from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Filtering:
    max_len: int = 512
    min_len: int = 60
    subset: Optional[str] = None
    allowed_oligomer: Tuple[str,] = ("monomeric",)
    max_helix_percent: float = 1.0
    max_loop_percent: float = 0.5
    min_beta_percent: float = -1.0
    rog_quantile: float = 0.96


@dataclass
class Data:
    csv_path: str = "./data/processed_pdb/metadata.csv"
    cluster_path: str = "./data/processed_pdb/clusters_30.txt"
    filtering: Filtering = Filtering()
    min_t: float = 0.01
    samples_per_eval_length: int = 4
    num_eval_lengths: int = 10
    num_t: int = 100


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
class Diffuser:
    diffuse_trans: bool = True
    diffuse_rot: bool = True
    r3: R3 = R3()
    so3: SO3 = SO3()


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
class Model:
    node_embed_size: int = 256
    edge_embed_size: int = 128
    dropout: float = 0.0
    embed: Embed = Embed()
    ipa: Ipa = Ipa()


@dataclass
class Experiment:
    name: str = "new_model_cluster_batch_warm_start_0"
    run_id: str = "mzrh22oe"
    use_ddp: bool = False
    log_freq: int = 1000
    batch_size: int = 256
    eval_batch_size: int = 4
    num_loader_workers: int = 5
    num_epoch: int = 10000000
    learning_rate: float = 0.0001
    max_squared_res: int = 600000
    prefetch_factor: int = 100
    use_gpu: bool = True
    num_gpus: int = 2
    sample_mode: str = "cluster_only_batch"
    wandb_dir: str = "./wandb/run-20230623_184455-az2r7aiw/files"
    use_wandb: bool = True
    ckpt_freq: int = 10000
    early_ckpt: bool = True
    warm_start: str = "ckpt/separate_rot_loss_05_big_batch_0/18D_06M_2023Y_21h_13m_18s"
    use_warm_start_conf: bool = False
    ckpt_dir: str = (
        "./ckpt/new_model_cluster_batch_warm_start_0/23D_06M_2023Y_18h_44m_54s"
    )
    trans_loss_weight: float = 1.0
    rot_loss_weight: float = 0.5
    rot_loss_t_upper: float = 1.0
    rot_loss_t_lower: float = 0.2
    rot_loss_clip: Optional[float] = None
    separate_rot_loss: bool = True
    trans_x0_threshold: float = 1.0
    coordinate_scaling: float = 0.1
    bb_atom_loss_weight: float = 1.0
    bb_atom_loss_t_filter: float = 0.25
    dist_mat_loss_weight: float = 1.0
    dist_mat_loss_t_filter: float = 0.25
    aux_loss_weight: float = 0.25
    eval_dir: str = (
        "./eval_outputs/new_model_cluster_batch_warm_start_0/23D_06M_2023Y_18h_44m_54s"
    )
    noise_scale: float = 1.0
    num_parameters: int = 17446190


@dataclass
class Config:
    data: Data = Data()
    diffuser: Diffuser = Diffuser()
    model: Model = Model()
    experiment: Experiment = Experiment()


@dataclass
class DiffusionParams:
    num_t: int = 500
    noise_scale: float = 0.1
    min_t: float = 0.01


@dataclass
class SamplerConfig:
    length: int = 100
    seq_per_sample: int = 8
    diffusion_params: DiffusionParams = DiffusionParams()
