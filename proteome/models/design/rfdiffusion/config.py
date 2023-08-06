from dataclasses import dataclass
from typing import List, Optional


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


@dataclass
class RoseTTAFoldModuleConfig:
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
    SE3_param_full: SE3Config = SE3Config()
    SE3_param_topk: SE3Config = SE3Config(
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
    freeze_track_motif: bool = False


@dataclass
class DiffuserConfig:
    T: int = 50
    b_0: float = 1e-2
    b_T: float = 7e-2
    schedule_type: str = "linear"
    so3_type: str = "igso3"
    crd_scale: float = 0.25
    partial_T: Optional[int] = None
    so3_schedule_type: str = "linear"
    min_b: float = 1.5
    max_b: float = 2.5
    min_sigma: float = 0.02
    max_sigma: float = 1.5
    chi_type: str = "interp"
    aa_decode_steps: int = 0


@dataclass
class PreprocessConfig:
    sidechain_input: bool = False
    motif_sidechain_input: bool = True
    d_t1d: int = 22
    d_t2d: int = 44
    predict_previous: bool = False
    new_self_cond: bool = False
    sequence_decode: bool = True
    seq_self_cond: bool = False
    provide_ss: bool = True
    seqalone: bool = True


@dataclass
class SeqDiffuserConfig:
    T: int = 200
    s_b0: float = 0.001
    s_bT: float = 0.1
    schedule_type: str = "cosine"
    seq_diff_type: str = None


@dataclass
class RFDiffusionModelConfig:
    model: RoseTTAFoldModuleConfig = RoseTTAFoldModuleConfig()
    diffuser: DiffuserConfig = DiffuserConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    seq_diffuser: Optional[SeqDiffuserConfig] = None


# Set all the model configs
BaseConfig = RFDiffusionModelConfig()
ActiveSiteConfig = BaseConfig
Base8Config = RFDiffusionModelConfig(
    seq_diffuser=SeqDiffuserConfig(),
    preprocess=PreprocessConfig(new_self_cond=True),
)
ComplexBaseConfig = RFDiffusionModelConfig(
    preprocess=PreprocessConfig(d_t1d=24),
)
ComplexFoldBaseConfig = RFDiffusionModelConfig(
    preprocess=PreprocessConfig(d_t1d=28, d_t2d=47),
)
InpaintSeqConfig = RFDiffusionModelConfig(
    preprocess=PreprocessConfig(d_t1d=24, provide_ss=False),
)
InpaintSeqFoldConfig = RFDiffusionModelConfig(
    preprocess=PreprocessConfig(d_t1d=28, d_t2d=47, provide_ss=False),
)
ComplexBetaConfig = InpaintSeqConfig


@dataclass
class DenoiserParams:
    noise_scale_ca: float = 1.0
    final_noise_scale_ca: float = 1.0
    ca_noise_schedule_type: str = "constant"
    noise_scale_frame: float = 1.0
    final_noise_scale_frame: float = 1.0
    frame_noise_schedule_type: str = "constant"


@dataclass
class PPIParams:
    hotspot_res: Optional[List[int]] = None


@dataclass
class SymmetryParams:
    # Symmetry to sample
    # Available symmetries:
    # - Cyclic symmetry (C_n) # call as c5
    # - Dihedral symmetry (D_n) # call as d5
    # - Tetrahedral symmetry # call as tetrahedral
    # - Octahedral symmetry # call as octahedral
    # - Icosahedral symmetry # call as icosahedral
    symmetry: str = "c2"
    model_only_neighbors: str = False
    output_prefix: str = "samples/c2"


@dataclass
class PotentialsParams:
    guiding_potentials: Optional[List] = None
    guide_scale: float = 10.0
    guide_decay: str = "constant"
    olig_inter_all: Optional[List] = None
    olig_intra_all: Optional[List] = None
    olig_custom_contact: Optional[List] = None
    substrate: Optional[List] = None


@dataclass
class ContigSettings:
    ref_idx: int = None
    hal_idx: int = None
    idx_rf: int = None
    inpaint_seq_tensor: Optional[List] = None


@dataclass
class ContigMap:
    contigs: str = None
    inpaint_seq: Optional[str] = None
    provide_seq: Optional[str] = None
    length: Optional[int] = None


@dataclass
class LoggingConfig:
    inputs: bool = False


@dataclass
class ScaffoldGuidedParams:
    scaffoldguided: bool = False
    target_pdb: bool = False
    target_path: Optional[str] = None
    scaffold_list: Optional[List[str]] = None
    scaffold_dir: Optional[str] = None
    sampled_insertion: int = 0
    sampled_N: int = 0
    sampled_C: int = 0
    ss_mask: int = 0
    systematic: bool = False
    target_ss: Optional[str] = None
    target_adj: Optional[str] = None
    mask_loops: bool = True
    contig_crop: Optional[List[str]] = None


@dataclass
class InferenceConfig:
    input_pdb: str = None
    num_designs: int = 10
    ckpt_override_path: str = None
    symmetry: str = None
    recenter: bool = True
    radius: float = 10.0
    model_only_neighbors: bool = False
    output_prefix: str = "samples/design"
    write_trajectory: bool = True
    scaffold_guided: bool = False
    model_runner: str = "SelfConditioning"
    align_motif: bool = True
    symmetric_self_cond: bool = True
    final_step: int = 1
    trb_save_ckpt_path: str = None
    schedule_directory_path: str = None
    model_directory_path: str = None


@dataclass
class RFDiffusionConfig:
    inference: InferenceConfig = InferenceConfig()
    contigmap: ContigMap = ContigMap()
    model: RoseTTAFoldModuleConfig = RoseTTAFoldModuleConfig()
    diffuser: DiffuserConfig = DiffuserConfig()
    denoiser: DenoiserParams = DenoiserParams()
    ppi: PPIParams = PPIParams()
    potentials: PotentialsParams = PotentialsParams()
    contig_settings: ContigSettings = ContigSettings()
    preprocess: PreprocessConfig = PreprocessConfig()
    logging: LoggingConfig = LoggingConfig()
    scaffoldguided: ScaffoldGuidedParams = ScaffoldGuidedParams()


"""
@dataclass
class InpaintSeqConfig(RFDiffuserConfig):
    if (
        conf.contigmap.inpaint_seq is not None
        or conf.contigmap.provide_seq is not None
    ):
        # use model trained for inpaint_seq
        if conf.contigmap.provide_seq is not None:
            # this is only used for partial diffusion
            assert (
                conf.diffuser.partial_T is not None
            ), "The provide_seq input is specifically for partial diffusion"
        if conf.scaffoldguided.scaffoldguided:
            self.ckpt_path = f"{model_directory}/InpaintSeq_Fold_ckpt.pt"
        else:
            self.ckpt_path = f"{model_directory}/InpaintSeq_ckpt.pt"
    elif (
        conf.ppi.hotspot_res is not None
        and conf.scaffoldguided.scaffoldguided is False
    ):
        # use complex trained model
        self.ckpt_path = f"{model_directory}/Complex_base_ckpt.pt"
    elif conf.scaffoldguided.scaffoldguided is True:
        # use complex and secondary structure-guided model
        self.ckpt_path = f"{model_directory}/Complex_Fold_base_ckpt.pt"
    else:
        # use default model
        self.ckpt_path = f"{model_directory}/Base_ckpt.pt"
# for saving in trb file:
assert (
    self._conf.inference.trb_save_ckpt_path is None
), "trb_save_ckpt_path is not the place to specify an input model. Specify in inference.ckpt_override_path"
self._conf["inference"]["trb_save_ckpt_path"] = self.ckpt_path
"""
