from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from proteome import protein


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

    def __post_init__(self):
        assert self.ca_noise_schedule_type in ["constant", "linear"]
        assert self.frame_noise_schedule_type in ["constant", "linear"]


@dataclass
class PPIParams:
    # Hotspot residues denoted by chain letter
    # and residue number (e.g. "A12", "B3")
    hotspot_res: Optional[List[str]] = None


@dataclass
class SymmetryParams:
    symmetry: Optional[str] = None
    model_only_neighbors: bool = False
    recenter: bool = True
    radius: float = 10.0
    symmetric_self_cond: bool = True

    def __post_init__(self):
        self.symmetry = self.symmetry.lower() if self.symmetry is not None else None
        if self.symmetry is not None:
            assert (self.symmetry in ["tetrahedral", "octahedral", "icosahedral"]) or (
                self.symmetry[0] in ["c", "d"] and self.symmetry[1:].isdigit()
            )


@dataclass
class PotentialsParams:
    guiding_potentials: Optional[List[str]] = None
    guide_scale: float = 10.0
    guide_decay: str = "constant"
    olig_inter_all: bool = False
    olig_intra_all: bool = False
    olig_custom_contact: Optional[List] = None
    substrate: Optional[str] = None

    def __post_init__(self):
        assert self.guide_decay in ["constant", "linear", "quadratic", "cubic"]


@dataclass
class ContigSettings:
    """
    This is currently unused in the Samplers but could be used in place of ContigMap
    to avoid the need for contig string syntax and parsing.
    """

    ref_idx: int = None
    hal_idx: int = None
    idx_rf: int = None
    inpaint_seq_tensor: Optional[torch.Tensor] = None


@dataclass
class ContigMap:
    contigs: str = None
    inpaint_seq: Optional[str] = None
    provide_seq: Optional[str] = None
    length: Optional[int] = None


@dataclass
class ScaffoldGuidedParams:
    target_structure: Optional[protein.Protein] = None
    scaffold_structure_list: List[protein.Protein] = None
    sampled_insertion: int = 0
    sampled_N: int = 0
    sampled_C: int = 0
    ss_mask: int = 0
    systematic: bool = False
    target_ss: bool = False
    target_adj: bool = False
    mask_loops: bool = True
    contig_crop: Optional[List[str]] = None


@dataclass
class InferenceParams:
    reference_structure: protein.Protein = None
    num_designs: int = 1
    align_motif: bool = True
    final_step: int = 1


@dataclass
class UnconditionalSamplerConfig:
    inference_params: InferenceParams = InferenceParams()
    contigmap_params: ContigMap = ContigMap()
    denoiser_params: DenoiserParams = DenoiserParams()
    potentials_params: PotentialsParams = PotentialsParams()
    symmetry_params: SymmetryParams = SymmetryParams()

    def __post_init__(self):
        assert self.inference_params.reference_structure is None
        assert self.contigmap_params.inpaint_seq is None
        assert self.contigmap_params.provide_seq is None
        assert self.contigmap_params.contigs is not None


@dataclass
class SelfConditioningSamplerConfig:
    inference_params: InferenceParams = InferenceParams()
    contigmap_params: ContigMap = ContigMap()
    denoiser_params: DenoiserParams = DenoiserParams()
    ppi_params: PPIParams = PPIParams()
    potentials_params: PotentialsParams = PotentialsParams()
    symmetry_params: SymmetryParams = SymmetryParams()


@dataclass
class ScaffoldedSamplerConfig:
    inference_params: InferenceParams = InferenceParams()
    contigmap_params: ContigMap = ContigMap()
    denoiser_params: DenoiserParams = DenoiserParams()
    ppi_params: PPIParams = PPIParams()
    potentials_params: PotentialsParams = PotentialsParams()
    symmetry_params: SymmetryParams = SymmetryParams()
    scaffoldguided_params: ScaffoldGuidedParams = ScaffoldGuidedParams()


SamplerConfigType = Union[
    UnconditionalSamplerConfig,
    SelfConditioningSamplerConfig,
    ScaffoldedSamplerConfig,
]
