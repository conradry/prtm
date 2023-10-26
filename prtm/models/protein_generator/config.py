from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from prtm import protein


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
    d_t1d: int = 24
    d_t2d: int = 44
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


BaseConfig = RoseTTAFoldModuleConfig()
ComplexConfig = RoseTTAFoldModuleConfig(d_t1d=29)


@dataclass
class HydrophobicBiasParams:
    hydrophobic_score: int = -10
    hydrophobic_loss_type: str = "complex"

    def __post_init__(self):
        self._potential_type = "hydrophobic"
        if self.hydrophobic_loss_type not in ["complex", "simple"]:
            raise ValueError(
                f"hydrophobic_loss_type must be one of ['complex', 'simple'], got {self.hydrophobic_loss_type}"
            )


@dataclass
class AACompositionalBiasParams:
    aa_composition: str = "W0.2"
    aa_spec: Optional[str] = None
    aa_weight: Optional[str] = None
    aa_weights_json: Optional[str] = None
    add_weight_every_n: int = 1
    frac_seq_to_weight: float = 0.0
    one_weight_per_position: bool = False

    def __post_init__(self):
        self._potential_type = "aa_bias"


@dataclass
class ChargeBiasParams:
    target_charge: int = -10
    charge_loss_type: str = "complex"
    target_pH: float = 7.4

    def __post_init__(self):
        self._potential_type = "charge"


PotentialParamType = Union[
    HydrophobicBiasParams, AACompositionalBiasParams, ChargeBiasParams
]


@dataclass
class PotentialsParams:
    potentials: Optional[List[PotentialParamType]] = None
    potential_scales: Optional[List[float]] = None

    def __post_init__(self):
        if self.potentials is not None:
            assert len(self.potentials) == len(self.potential_scales)


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
    contigs: Optional[str] = None
    inpaint_seq: Optional[str] = None
    provide_seq: Optional[str] = None
    length: Optional[int] = None


@dataclass
class StructureBiasParams:
    helix_bias: float = 0.0
    strand_bias: float = 0.0
    loop_bias: float = 0.0


@dataclass
class HotspotParams:
    # Hotspot residues denoted by chain letter
    # and residue number (e.g. "A12", "B3")
    hotspot_res: Optional[List[str]] = None


@dataclass
class SymmetryParams:
    symmetry: int = 1
    symmetry_cap: int = 0
    predict_symmetric: bool = False


@dataclass
class SecondaryStructureParams:
    dssp_structure: Optional[protein.ProteinBase] = None
    secondary_structure: str = None

    def __post_init__(self):
        assert not (
            (self.secondary_structure != None) and (self.dssp_structure != None)
        ), f"You are attempting to provide both dssp_structure and/or secondary_secondary structure, please choose one or the other"
        if self.secondary_structure is not None:
            assert all([x in "XHEL" for x in self.secondary_structure])


@dataclass
class DiffuserParams:
    T: int = 25
    schedule: str = "sqrt"
    sample_distribution: str = "normal"
    sample_distribution_gmm_means: Tuple[float] = (0,)
    sample_distribution_gmm_variances: Tuple[float] = (1,)

    def __post_init__(self):
        assert self.schedule in ["sqrt", "linear", "none"]
        assert self.sample_distribution in ["normal", "uniform", "none"]


@dataclass
class InferenceConfig:
    contigmap_params: ContigMap
    sequence: Optional[str] = None
    reference_structure: Optional[protein.Protein27] = None

    potentials_params: PotentialsParams = PotentialsParams()
    structure_bias_params: StructureBiasParams = StructureBiasParams()
    hotspot_params: HotspotParams = HotspotParams()
    diffuser_params: DiffuserParams = DiffuserParams()
    symmetry_params: SymmetryParams = SymmetryParams()
    secondary_structure_params: SecondaryStructureParams = SecondaryStructureParams()

    # Other inference parameters
    clamp_seqout: bool = False
    softmax_seqout: bool = False
    n_cycle: int = 4
    sampling_temp: float = 1.0
    scheduled_str_cond: bool = False
    struc_cond_sc: bool = False
    tmpl_conf: float = 1

    def __post_init__(self):
        if self.reference_structure is not None:
            self.reference_structure = self.reference_structure.to_protein27()
