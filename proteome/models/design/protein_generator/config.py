from dataclasses import dataclass
from typing import List, Optional

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
    d_t1d: int = 22
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
    freeze_track_motif: bool = False
    input_seq_onehot: bool = False


BaseConfig = RoseTTAFoldModuleConfig()
ComplexConfig = RoseTTAFoldModuleConfig(d_t1d=29)


@dataclass
class InferenceConfig:
    F: int = 1
    T: int = 25
    aa_composition: str = "W0.2"
    aa_spec: Optional[str] = None
    aa_weight: Optional[str] = None
    aa_weights_json: Optional[str] = None
    add_weight_every_n: int = 1
    argmax_seq: bool = False
    cautious: bool = False
    checkpoint: str = "./SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt"
    clamp_seqout: bool = False
    contigs: List[str] = ["0"]
    d_t1d: int = 24
    dssp_pdb: Optional[str] = None
    dump_all: bool = False
    dump_npz: bool = False
    dump_pdb: bool = True
    dump_trb: bool = True
    frac_seq_to_weight: float = 0.0
    hal_idx: Optional[str] = None
    helix_bias: float = 0.0
    hotspots: Optional[str] = None
    idx_rf: Optional[str] = None
    inpaint_seq: Optional[str] = None
    inpaint_seq_tensor: Optional[str] = None
    inpaint_str: Optional[str] = None
    inpaint_str_tensor: Optional[str] = None
    input_json: Optional[str] = None
    length: Optional[int] = None
    loop_bias: float = 0.0
    loop_design: bool = False
    min_decoding_distance: int = 15
    multi_templates: Optional[str] = None
    multi_tmpl_conf: Optional[str] = None
    n_cycle: int = 4
    noise_schedule: str = "sqrt"
    num_designs: int = 500
    one_weight_per_position: bool = False
    out: str = "./"
    pdb: Optional[str] = None
    potential_scale: str = ""
    ref_idx: Optional[str] = None
    sampling_temp: float = 1.0
    save_all_steps: bool = False
    save_best_plddt: bool = True
    save_seqs: bool = False
    scheduled_str_cond: bool = False
    secondary_structure: Optional[str] = None
    softmax_seqout: bool = False
    start_num: int = 0
    strand_bias: float = 0.0
    struc_cond_sc: bool = False
    symmetry: int = 1
    symmetry_cap: int = 0
    temperature: float = 0.1
    tmpl_conf: str = "1"
    trb: Optional[str] = None
    sample_distribution: str = "normal"
    sample_distribution_gmm_means: List[float] = [0]
    sample_distribution_gmm_variances: List[float] = [1]
    target_charge: int = -10
    charge_loss_type: str = "complex"
    target_pH: float = 7.4
    hydrophobic_score: int = -10
    hydrophobic_loss_type: str = "complex"
    save_args: bool = True
    potentials: str = ""
    sequence: str = "XXXXXXXXXXXXXXXXPEPSEQXXXXXXXXXXXXXXXX"
    sampler: str = "default"
    predict_symmetric: str = "false"
