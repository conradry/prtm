from dataclasses import dataclass


@dataclass
class InferenceConfig:
    input_pdb: str = None
    num_designs: int = 10
    design_startnum: int = 0
    ckpt_override_path: str = None
    symmetry: str = None
    recenter: bool = True
    radius: float = 10.0
    model_only_neighbors: bool = False
    output_prefix: str = "samples/design"
    write_trajectory: bool = True
    scaffold_guided: bool = False
    model_runner: str = "SelfConditioning"
    cautious: bool = True
    align_motif: bool = True
    symmetric_self_cond: bool = True
    final_step: int = 1
    deterministic: bool = False
    trb_save_ckpt_path: str = None
    schedule_directory_path: str = None
    model_directory_path: str = None


@dataclass
class ContigMap:
    contigs: list = None
    inpaint_seq: str = None
    provide_seq: str = None
    length: int = None


@dataclass
class SE3Config:
    num_layers: int = 1
    num_channels: int = 32
    num_degrees: int = 2
    n_heads: int = 4
    div: int = 4
    l0_in_features: int = 8
    l0_out_features: int = 8
    l1_in_features: int = 3
    l1_out_features: int = 2
    num_edge_features: int = 32


SE3_param_full = SE3Config()
SE3_param_topk = SE3Config(
    num_layers=1,
    num_channels=32,
    num_degrees=2,
    n_heads=4,
    div=4,
    l0_in_features=64,
    l0_out_features=64,
    l1_in_features=3,
    l1_out_features=2,
    num_edge_features=64,
)


@dataclass
class ModelConfig:
    n_extra_block: int = 4
    n_main_block: int = 32
    n_ref_block: int = 4
    d_msa: int = 256
    d_msa_full: int = 64
    d_pair: int = 128
    d_templ: int = 64
    n_head_msa: int = 8
    n_head_pair: int = 4
    n_head_templ: int = 4
    d_hidden: int = 32
    d_hidden_templ: int = 32
    p_drop: float = 0.15
    SE3_param_full: SE3Config = SE3_param_full
    SE3_param_topk: SE3Config = SE3_param_topk
    freeze_track_motif: bool = False
    use_motif_timestep: bool = False


@dataclass
class DiffusionConfig:
    T: int = 50
    b_0: float = 1e-2
    b_T: float = 7e-2
    schedule_type: str = "linear"
    so3_type: str = "igso3"
    crd_scale: float = 0.25
    partial_T: int = None
    so3_schedule_type: str = "linear"
    min_b: float = 1.5
    max_b: float = 2.5
    min_sigma: float = 0.02
    max_sigma: float = 1.5


@dataclass
class DenoiserConfig:
    noise_scale_ca: float = 1.0
    final_noise_scale_ca: float = 1.0
    ca_noise_schedule_type: str = "constant"
    noise_scale_frame: float = 1.0
    final_noise_scale_frame: float = 1.0
    frame_noise_schedule_type: str = "constant"


@dataclass
class PPIConfig:
    hotspot_res: list = None


@dataclass
class PotentialsConfig:
    guiding_potentials: list = None
    guide_scale: float = 10.0
    guide_decay: str = "constant"
    olig_inter_all: list = None
    olig_intra_all: list = None
    olig_custom_contact: list = None
    substrate: list = None


@dataclass
class ContigSettings:
    ref_idx: int = None
    hal_idx: int = None
    idx_rf: int = None
    inpaint_seq_tensor: list = None


@dataclass
class PreprocessConfig:
    sidechain_input: bool = False
    motif_sidechain_input: bool = True
    d_t1d: int = 22
    d_t2d: int = 44
    prob_self_cond: float = 0.0
    str_self_cond: bool = False
    predict_previous: bool = False


@dataclass
class LoggingConfig:
    inputs: bool = False


@dataclass
class ScaffoldGuidedConfig:
    scaffoldguided: bool = False
    target_pdb: bool = False
    target_path: str = None
    scaffold_list: list = None
    scaffold_dir: str = None
    sampled_insertion: int = 0
    sampled_N: int = 0
    sampled_C: int = 0
    ss_mask: int = 0
    systematic: bool = False
    target_ss: str = None
    target_adj: str = None
    mask_loops: bool = True
    contig_crop: list = None


@dataclass
class RFDiffusionConfig:
    inference: InferenceConfig = InferenceConfig()
    contigmap: ContigMap = ContigMap()
    model: ModelConfig = ModelConfig()
    diffuser: DiffusionConfig = DiffusionConfig()
    denoiser: DenoiserConfig = DenoiserConfig()
    ppi: PPIConfig = PPIConfig()
    potentials: PotentialsConfig = PotentialsConfig()
    contig_settings: ContigSettings = ContigSettings()
    potentials: PotentialsConfig = PotentialsConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    logging: LoggingConfig = LoggingConfig()
    scaffoldguided: ScaffoldGuidedConfig = ScaffoldGuidedConfig()
