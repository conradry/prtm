from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

SHAPE_SCHEMA = {
    "aatype": [NUM_RES],
    "all_atom_mask": [NUM_RES, None],
    "all_atom_positions": [NUM_RES, None, None],
    "alt_chi_angles": [NUM_RES, None],
    "atom14_alt_gt_exists": [NUM_RES, None],
    "atom14_alt_gt_positions": [NUM_RES, None, None],
    "atom14_atom_exists": [NUM_RES, None],
    "atom14_atom_is_ambiguous": [NUM_RES, None],
    "atom14_gt_exists": [NUM_RES, None],
    "atom14_gt_positions": [NUM_RES, None, None],
    "atom37_atom_exists": [NUM_RES, None],
    "backbone_rigid_mask": [NUM_RES],
    "backbone_rigid_tensor": [NUM_RES, None, None],
    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
    "chi_angles_sin_cos": [NUM_RES, None, None],
    "chi_mask": [NUM_RES, None],
    "extra_deletion_value": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_has_deletion": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_row_mask": [NUM_EXTRA_SEQ],
    "is_distillation": [],
    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
    "msa_row_mask": [NUM_MSA_SEQ],
    "no_recycling_iters": [],
    "pseudo_beta": [NUM_RES, None],
    "pseudo_beta_mask": [NUM_RES],
    "residue_index": [NUM_RES],
    "residx_atom14_to_atom37": [NUM_RES, None],
    "residx_atom37_to_atom14": [NUM_RES, None],
    "resolution": [],
    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
    "rigidgroups_group_exists": [NUM_RES, None],
    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
    "rigidgroups_gt_exists": [NUM_RES, None],
    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
    "seq_length": [],
    "seq_mask": [NUM_RES],
    "target_feat": [NUM_RES, None],
    "template_aatype": [NUM_TEMPLATES, NUM_RES],
    "template_all_atom_mask": [NUM_TEMPLATES, NUM_RES, None],
    "template_all_atom_positions": [
        NUM_TEMPLATES,
        NUM_RES,
        None,
        None,
    ],
    "template_alt_torsion_angles_sin_cos": [
        NUM_TEMPLATES,
        NUM_RES,
        None,
        None,
    ],
    "template_backbone_rigid_mask": [NUM_TEMPLATES, NUM_RES],
    "template_backbone_rigid_tensor": [
        NUM_TEMPLATES,
        NUM_RES,
        None,
        None,
    ],
    "template_mask": [NUM_TEMPLATES],
    "template_pseudo_beta": [NUM_TEMPLATES, NUM_RES, None],
    "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_RES],
    "template_sum_probs": [NUM_TEMPLATES, None],
    "template_torsion_angles_mask": [
        NUM_TEMPLATES,
        NUM_RES,
        None,
    ],
    "template_torsion_angles_sin_cos": [
        NUM_TEMPLATES,
        NUM_RES,
        None,
        None,
    ],
    "true_msa": [NUM_MSA_SEQ, NUM_RES],
    "use_clamped_fape": [],
}


@dataclass
class Features:
    aatype: Optional[np.ndarray] = None
    all_atom_mask: Optional[np.ndarray] = None
    all_atom_positions: Optional[np.ndarray] = None
    alt_chi_angles: Optional[np.ndarray] = None
    atom14_alt_gt_exists: Optional[np.ndarray] = None
    atom14_alt_gt_positions: Optional[np.ndarray] = None
    atom14_atom_exists: Optional[np.ndarray] = None
    atom14_atom_is_ambiguous: Optional[np.ndarray] = None
    atom14_gt_exists: Optional[np.ndarray] = None
    atom14_gt_positions: Optional[np.ndarray] = None
    atom37_atom_exists: Optional[np.ndarray] = None
    backbone_rigid_mask: Optional[np.ndarray] = None
    backbone_rigid_tensor: Optional[np.ndarray] = None
    between_segment_residues: Optional[np.ndarray] = None
    bert_mask: Optional[np.ndarray] = None
    chi_angles_sin_cos: Optional[np.ndarray] = None
    chi_mask: Optional[np.ndarray] = None
    deletion_matrix_int: Optional[np.ndarray] = None
    extra_deletion_value: Optional[np.ndarray] = None
    extra_has_deletion: Optional[np.ndarray] = None
    extra_msa: Optional[np.ndarray] = None
    extra_msa_mask: Optional[np.ndarray] = None
    extra_msa_row_mask: Optional[np.ndarray] = None
    is_distillation: Optional[np.ndarray] = None
    msa: Optional[np.ndarray] = None
    msa_feat: Optional[np.ndarray] = None
    msa_mask: Optional[np.ndarray] = None
    msa_row_mask: Optional[np.ndarray] = None
    no_recycling_iters: Optional[np.ndarray] = None
    num_alignments: Optional[np.ndarray] = None
    pseudo_beta: Optional[np.ndarray] = None
    pseudo_beta_mask: Optional[np.ndarray] = None
    residue_index: Optional[np.ndarray] = None
    residx_atom14_to_atom37: Optional[np.ndarray] = None
    residx_atom37_to_atom14: Optional[np.ndarray] = None
    resolution: Optional[np.ndarray] = None
    rigidgroups_alt_gt_frames: Optional[np.ndarray] = None
    rigidgroups_group_exists: Optional[np.ndarray] = None
    rigidgroups_group_is_ambiguous: Optional[np.ndarray] = None
    rigidgroups_gt_exists: Optional[np.ndarray] = None
    rigidgroups_gt_frames: Optional[np.ndarray] = None
    seq_length: Optional[np.ndarray] = None
    seq_mask: Optional[np.ndarray] = None
    target_feat: Optional[np.ndarray] = None
    template_aatype: Optional[np.ndarray] = None
    template_all_atom_mask: Optional[np.ndarray] = None
    template_all_atom_positions: Optional[np.ndarray] = None
    template_domain_names: Optional[np.ndarray] = None
    template_alt_torsion_angles_sin_cos: Optional[np.ndarray] = None
    template_backbone_rigid_mask: Optional[np.ndarray] = None
    template_backbone_rigid_tensor: Optional[np.ndarray] = None
    template_mask: Optional[np.ndarray] = None
    template_pseudo_beta: Optional[np.ndarray] = None
    template_pseudo_beta_mask: Optional[np.ndarray] = None
    template_sum_probs: Optional[np.ndarray] = None
    template_torsion_angles_mask: Optional[np.ndarray] = None
    template_torsion_angles_sin_cos: Optional[np.ndarray] = None
    true_msa: Optional[np.ndarray] = None
    use_clamped_fape: Optional[np.ndarray] = None


@dataclass
class MaskedMSA:
    profile_prob: float = 0.1
    same_prob: float = 0.1
    uniform_prob: float = 0.1


@dataclass
class CommonData:
    masked_msa: MaskedMSA = MaskedMSA()
    max_recycling_iters: int = 3
    msa_cluster_features: bool = True
    reduce_msa_clusters_by_max_templates: bool = False
    resample_msa_in_recycling: bool = True
    template_features: Tuple[str, ...] = (
        "template_all_atom_positions",
        "template_sum_probs",
        "template_aatype",
        "template_all_atom_mask",
    )
    unsupervised_features: Tuple[str, ...] = (
        "aatype",
        "residue_index",
        "msa",
        "num_alignments",
        "seq_length",
        "between_segment_residues",
        "deletion_matrix",
        "no_recycling_iters",
    )
    use_templates: bool = True
    use_template_torsion_angles: bool = True


@dataclass
class PredictData:
    subsample_templates: bool = False
    masked_msa_replace_fraction: float = 0.15
    max_msa_clusters: int = 512
    max_extra_msa: int = 1024
    max_template_hits: int = 4
    max_templates: int = 4
    crop: bool = False
    crop_size: Optional[int] = None
    uniform_recycling: bool = False

@dataclass
class DataModule:
    use_small_bfd: bool = False


@dataclass
class DataConfig:
    common: CommonData = CommonData()
    predict: PredictData = PredictData()
    data_module: DataModule = DataModule()


@dataclass
class GlobalsConfig:
    blocks_per_ckpt: Optional[int] = None
    c_z: int = 128
    c_m: int = 256
    c_t: int = 64
    c_e: int = 64
    c_s: int = 384
    chunk_size: int = 4
    eps: float = 1e-8
    use_lma: bool = False
    use_flash: bool = False
    offload_inference: bool = False


@dataclass
class InputEmbedderConfig:
    tf_dim: int = 22
    msa_dim: int = 49
    c_z: float = 128
    c_m: float = 256
    relpos_k: int = 32


@dataclass
class RecyclingEmbedderConfig:
    c_z: int = 128
    c_m: int = 256
    min_bin: float = 3.25
    max_bin: float =  20.75
    no_bins: int =  15
    inf: float = 1e8


@dataclass
class TemplateAngleEmbedder:
    c_in: int = 57
    c_out: int = 256


@dataclass
class TemplatePairEmbedder:
    c_in: int = 88
    c_out: int = 64


@dataclass
class TemplatePairStack:
    c_t: int = 64
    c_hidden_tri_att: int = 16
    c_hidden_tri_mul: int = 64
    no_blocks: int = 2
    no_heads: int = 4
    pair_transition_n: int = 2
    dropout_rate: float = 0.25
    blocks_per_ckpt: Optional[int] = None
    tune_chunk_size: bool = True
    inf: float = 1e9


@dataclass
class TemplatePointwiseAttention:
    c_t: int = 64
    c_z: int = 128
    c_hidden: int = 16
    no_heads: int = 4
    inf: float = 1e5


@dataclass
class Distogram:
    min_bin: float = 3.25
    max_bin: float = 50.75
    no_bins: int = 39


@dataclass
class TemplateConfig:
    distogram: Distogram = Distogram()
    template_angle_embedder: TemplateAngleEmbedder = TemplateAngleEmbedder()
    template_pair_embedder: TemplatePairEmbedder = TemplatePairEmbedder()
    template_pair_stack: TemplatePairStack = TemplatePairStack()
    template_pointwise_attention: TemplatePointwiseAttention = (
        TemplatePointwiseAttention()
    )
    inf: float = 1e9
    eps: float = 1e-6
    enabled: bool = True
    embed_angles: bool = False
    use_unit_vector: bool = False
    average_templates: bool = False
    offload_templates: bool = False


@dataclass
class ExtraMsaEmbedder:
    c_in: int = 25
    c_out: int = 64


@dataclass
class ExtraMsaStack:
    c_m: int = 64
    c_z: int = 128
    c_hidden_msa_att: int = 8
    c_hidden_opm: int = 32
    c_hidden_mul: int = 128
    c_hidden_pair_att: int = 32
    no_heads_msa: int = 8
    no_heads_pair: int = 4
    no_blocks: int = 4
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    clear_cache_between_blocks: bool = False
    tune_chunk_size: bool = True
    inf: float = 1e9
    eps: float = 1e-8
    ckpt: bool = False


@dataclass
class ExtraMsaConfig:
    extra_msa_embedder: ExtraMsaEmbedder = ExtraMsaEmbedder()
    extra_msa_stack: ExtraMsaStack = ExtraMsaStack()
    enabled: bool = True


@dataclass
class EvoformerStack:
    c_m: int = 256
    c_z: int = 128
    c_hidden_msa_att: int = 32
    c_hidden_opm: int = 32
    c_hidden_mul: int = 128
    c_hidden_pair_att: int = 32
    c_s: int = 384
    no_heads_msa: int = 8
    no_heads_pair: int = 4
    no_blocks: int = 48
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    blocks_per_ckpt: Optional[int] = None
    clear_cache_between_blocks: bool = False
    tune_chunk_size: bool = True
    inf: float = 1e9
    eps: float = 1e-8


@dataclass
class StructureModule:
    c_s: int = 384
    c_z: int = 128
    c_ipa: int = 16
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.1
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5


@dataclass
class HeadsLddt:
    no_bins: int = 50
    c_in: int = 384
    c_hidden: int = 128


@dataclass
class HeadsDistogram:
    c_z: int = 128
    no_bins: int = 64


@dataclass
class HeadsTm:
    c_z: int = 128
    no_bins: int = 64
    enabled: bool = False


@dataclass
class HeadsMaskedMsa:
    c_m: int = 256
    c_out: int = 23


@dataclass
class HeadsExperimentallyResolved:
    c_s: int = 384
    c_out: int = 37


@dataclass
class HeadsConfig:
    lddt: HeadsLddt = HeadsLddt()
    distogram: HeadsDistogram = HeadsDistogram()
    tm: HeadsTm = HeadsTm()
    masked_msa: HeadsMaskedMsa = HeadsMaskedMsa()
    experimentally_resolved: HeadsExperimentallyResolved = HeadsExperimentallyResolved()


@dataclass
class ModelConfig:
    _mask_trans: bool = False
    input_embedder: InputEmbedderConfig = InputEmbedderConfig()
    recycling_embedder: RecyclingEmbedderConfig = RecyclingEmbedderConfig()
    template: TemplateConfig = TemplateConfig()
    extra_msa: ExtraMsaConfig = ExtraMsaConfig()
    evoformer_stack: EvoformerStack = EvoformerStack()
    structure_module: StructureModule = StructureModule()
    heads: HeadsConfig = HeadsConfig()


@dataclass
class RelaxConfig:
    max_iterations: int = 0
    tolerance: float = 2.39
    stiffness: float = 10.0
    max_outer_iterations: int = 20
    exclude_residues: Tuple[int, ...] = ()


@dataclass
class DistogramLoss:
    min_bin: float = 2.3125
    max_bin: float = 21.6875
    no_bins: int = 64
    eps: float = 1e8
    weight: float = 0.3


@dataclass
class ExperimentallyResolvedLoss:
    eps: float = 1e-8
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    weight: float = 0.0


@dataclass
class BackboneFAPELoss:
    clamp_distance: float = 10.0
    loss_unit_distance: float = 10.0
    weight: float = 0.5


@dataclass
class SidechainFAPELoss:
    clamp_distance: float = 10.0
    length_scale: float = 10.0
    weight: float = 0.5


@dataclass
class FAPELoss:
    backbone: BackboneFAPELoss = BackboneFAPELoss()
    sidechain: SidechainFAPELoss = SidechainFAPELoss()
    eps: float = 1e-4
    weight: float = 1.0


@dataclass
class PLDDTLoss:
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    cutoff: float = 15.0
    no_bins: int = 50
    eps: float = 1e-10
    weight: float = 0.01


@dataclass
class MaskedMSALoss:
    eps: float = 1e-8
    weight: float = 2.0


@dataclass
class SupervisedChiLoss:
    chi_weight: float = 0.5
    angle_norm_weight: float = 0.01
    eps: float = 1e-6
    weight: float = 1.0


@dataclass
class ViolationLoss:
    violation_tolerance_factor: float = 12.0
    clash_overlap_tolerance: float = 1.5
    eps: float = 1e-6
    weight: float = 0.0


@dataclass
class TMLoss:
    max_bin: int = 31
    no_bins: int = 64
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    eps: float = 1e-8
    weight: float = 0.0
    enabled: bool = False


@dataclass
class LossConfig:
    distogram: DistogramLoss = DistogramLoss()
    experimentally_resolved: ExperimentallyResolvedLoss = ExperimentallyResolvedLoss()
    fape: FAPELoss = FAPELoss()
    plddt_loss: PLDDTLoss = PLDDTLoss()
    masked_msa: MaskedMSALoss = MaskedMSALoss()
    supervised_chi: SupervisedChiLoss = SupervisedChiLoss()
    violation: ViolationLoss = ViolationLoss()
    tm: TMLoss = TMLoss()
    eps: float = 1e8


@dataclass
class EmaConfig:
    decay: float = 0.999


@dataclass
class OpenFoldConfig:
    data: DataConfig = DataConfig()
    globals: GlobalsConfig = GlobalsConfig()
    model: ModelConfig = ModelConfig()
    relax: RelaxConfig = RelaxConfig()
    loss: LossConfig = LossConfig()
    ema: EmaConfig = EmaConfig()

    def __post_init__(self):
        if self.model.template.average_templates and self.globals.use_lma:
            raise ValueError(
                "Only one of use_lma and average_templates may be set at a time"
            )
        if self.model.template.offload_templates and self.globals.use_flash:
            raise ValueError(
                "Only one of use_flash and offload_templates may be set at a time"
            )

        if self.globals.use_flash and not self._is_flash_attention_installed():
            raise ValueError("use_flash requires that FlashAttention is installed")

    def _is_flash_attention_installed(self):
        try:
            import flash_attn

            return True
        except ImportError:
            return False


finetuning_config = OpenFoldConfig(
    loss=LossConfig(
        violation=ViolationLoss(
            weight=1.0,
        ),
        experimentally_resolved=ExperimentallyResolvedLoss(
            weight=0.01,
        ),
    ),
)

finetuning_ptm_config = OpenFoldConfig(
    model=ModelConfig(
        heads=HeadsConfig(
            tm=HeadsTm(
                enabled=True,
            ),
        ),
    ),
    loss=LossConfig(
        violation=ViolationLoss(
            weight=1.0,
        ),
        experimentally_resolved=ExperimentallyResolvedLoss(
            weight=0.01,
        ),
        tm=TMLoss(
            weight=0.1,
        ),
    ),
)

finetuning_no_templ_config = OpenFoldConfig(
    model=ModelConfig(
        template=TemplateConfig(
            enabled=False,
        ),
    ),
    loss=LossConfig(
        violation=ViolationLoss(
            weight=1.0,
        ),
        experimentally_resolved=ExperimentallyResolvedLoss(
            weight=0.01,
        ),
    ),
)

finetuning_no_templ_ptm_config = OpenFoldConfig(
    model=ModelConfig(
        template=TemplateConfig(
            enabled=False,
        ),
        heads=HeadsConfig(
            tm=HeadsTm(
                enabled=True,
            ),
        ),
    ),
    loss=LossConfig(
        violation=ViolationLoss(
            weight=1.0,
        ),
        experimentally_resolved=ExperimentallyResolvedLoss(
            weight=0.01,
        ),
        tm=TMLoss(
            weight=0.1,
        ),
    ),
)
