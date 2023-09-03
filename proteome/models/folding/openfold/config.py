from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

"""
@dataclass
class CommonFeatures:
    aatype: np.ndarray
    all_atom_mask: np.ndarray
    all_atom_positions: np.ndarray
    alt_chi_angles: np.ndarray
    atom14_alt_gt_exists: np.ndarray
    atom14_alt_gt_positions: np.ndarray
    atom14_atom_exists: np.ndarray
    atom14_atom_is_ambiguous: np.ndarray
    atom14_gt_exists: np.ndarray
    atom14_gt_positions: np.ndarray
    atom37_atom_exists: np.ndarray
    backbone_rigid_mask: np.ndarray
    backbone_rigid_tensor: np.ndarray
    bert_mask: np.ndarray
    chi_angles_sin_cos: np.ndarray
    chi_mask: np.ndarray
    extra_deletion_value: np.ndarray
    extra_has_deletion: np.ndarray
    extra_msa: np.ndarray
    extra_msa_mask: np.ndarray
    extra_msa_row_mask: np.ndarray
    is_distillation: np.ndarray
    msa_feat: np.ndarray
    msa_mask: np.ndarray
    msa_row_mask: np.ndarray
    no_recycling_iters: np.ndarray
    pseudo_beta: np.ndarray
    pseudo_beta_mask: np.ndarray
    residue_index: np.ndarray
    residx_atom14_to_atom37: np.ndarray
    residx_atom37_to_atom14: np.ndarray
    resolution: np.ndarray
    rigidgroups_alt_gt_frames: np.ndarray
    rigidgroups_group_exists: np.ndarray
    rigidgroups_group_is_ambiguous: np.ndarray
    rigidgroups_gt_exists: np.ndarray
    rigidgroups_gt_frames: np.ndarray
    seq_length: np.ndarray
    seq_mask: np.ndarray
    target_feat: np.ndarray
    template_aatype: np.ndarray
    template_all_atom_mask: np.ndarray
    template_all_atom_positions: np.ndarray
    template_alt_torsion_angles_sin_cos: np.ndarray
    template_backbone_rigid_mask: np.ndarray
    template_backbone_rigid_tensor: np.ndarray
    template_mask: np.ndarray
    template_pseudo_beta: np.ndarray
    template_pseudo_beta_mask: np.ndarray
    template_sum_probs: np.ndarray
    template_torsion_angles_mask: np.ndarray
    template_torsion_angles_sin_cos: np.ndarray
    true_msa: np.ndarray
    use_clamped_fape: np.ndarray
"""
common_features = (
    "aatype",
    "all_atom_mask",
    "all_atom_positions",
    "alt_chi_angles",
    "atom14_alt_gt_exists",
    "atom14_alt_gt_positions",
    "atom14_atom_exists",
    "atom14_atom_is_ambiguous",
    "atom14_gt_exists",
    "atom14_gt_positions",
    "atom37_atom_exists",
    "backbone_rigid_mask",
    "backbone_rigid_tensor",
    "bert_mask",
    "chi_angles_sin_cos",
    "chi_mask",
    "extra_deletion_value",
    "extra_has_deletion",
    "extra_msa",
    "extra_msa_mask",
    "extra_msa_row_mask",
    "is_distillation",
    "msa_feat",
    "msa_mask",
    "msa_row_mask",
    "no_recycling_iters",
    "pseudo_beta",
    "pseudo_beta_mask",
    "residue_index",
    "residx_atom14_to_atom37",
    "residx_atom37_to_atom14",
    "resolution",
    "rigidgroups_alt_gt_frames",
    "rigidgroups_group_exists",
    "rigidgroups_group_is_ambiguous",
    "rigidgroups_gt_exists",
    "rigidgroups_gt_frames",
    "seq_length",
    "seq_mask",
    "target_feat",
    "template_aatype",
    "template_all_atom_mask",
    "template_all_atom_positions",
    "template_alt_torsion_angles_sin_cos",
    "template_backbone_rigid_mask",
    "template_backbone_rigid_tensor",
    "template_mask",
    "template_pseudo_beta",
    "template_pseudo_beta_mask",
    "template_sum_probs",
    "template_torsion_angles_mask",
    "template_torsion_angles_sin_cos",
    "true_msa",
    "use_clamped_fape",
)


@dataclass
class MaskedMSA:
    profile_prob: float = 0.1
    same_prob: float = 0.1
    uniform_prob: float = 0.1


@dataclass
class CommonData:
    # feat: CommonFeatures
    feat: Tuple[str, ...] = common_features
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
class SupervisedData:
    clamp_prob: float = 0.9
    supervised_features: Tuple[str, ...] = (
        "all_atom_mask",
        "all_atom_positions",
        "resolution",
        "use_clamped_fape",
        "is_distillation",
    )


@dataclass
class PredictData:
    fixed_size: bool = True
    subsample_templates: bool = False
    masked_msa_replace_fraction: float = 0.15
    max_msa_clusters: int = 512
    max_extra_msa: int = 1024
    max_template_hits: int = 4
    max_templates: int = 4
    crop: bool = False
    crop_size: Tuple[int, int] = None
    supervised: bool = False
    uniform_recycling: bool = False


@dataclass
class EvalData:
    fixed_size: bool = True
    subsample_templates: bool = False
    masked_msa_replace_fraction: float = 0.15
    max_msa_clusters: int = 128
    max_extra_msa: int = 1024
    max_template_hits: int = 4
    max_templates: int = 4
    crop: bool = False
    crop_size: Tuple[int, int] = None
    supervised: bool = True
    uniform_recycling: bool = False


@dataclass
class TrainData:
    fixed_size: bool = True
    subsample_templates: bool = True
    masked_msa_replace_fraction: float = 0.15
    max_msa_clusters: int = 128
    max_extra_msa: int = 1024
    max_template_hits: int = 4
    max_templates: int = 4
    shuffle_top_k_prefiltered: int = 20
    crop: bool = True
    crop_size: int = 256
    supervised: bool = True
    clamp_prob: float = 0.9
    max_distillation_msa_clusters: int = 1000
    uniform_recycling: bool = True
    distillation_prob: float = 0.75


@dataclass
class DataModule:
    use_small_bfd: bool = False


@dataclass
class DataConfig:
    common: CommonData = CommonData()
    supervised: SupervisedData = SupervisedData()
    predict: PredictData = PredictData()
    eval: EvalData = EvalData()
    train: TrainData = TrainData()
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
    data=DataConfig(
        train=TrainData(
            crop_size=384,
            max_extra_msa=5120,
            max_msa_clusters=512,
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

finetuning_ptm_config = OpenFoldConfig(
    data=DataConfig(
        train=TrainData(
            crop_size=384,
            max_extra_msa=5120,
            max_msa_clusters=512,
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
    data=DataConfig(
        train=TrainData(
            crop_size=384,
            max_extra_msa=5120,
            max_msa_clusters=512,
        ),
    ),
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
    data=DataConfig(
        train=TrainData(
            crop_size=384,
            max_extra_msa=5120,
            max_msa_clusters=512,
        ),
    ),
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
