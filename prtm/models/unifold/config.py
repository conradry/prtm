import copy
from dataclasses import _MISSING_TYPE, dataclass, field, fields, is_dataclass
from typing import Any, List, Optional, Tuple

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
    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
    "chi_angles_sin_cos": [NUM_RES, None, None],
    "chi_mask": [NUM_RES, None],
    "extra_msa_deletion_value": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_has_deletion": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_row_mask": [NUM_EXTRA_SEQ],
    "frame_mask": [NUM_RES],
    "is_distillation": [],
    "msa_chains": [NUM_MSA_SEQ, None],
    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
    "msa_row_mask": [NUM_MSA_SEQ],
    "num_asym": [],
    "num_recycling_iters": [],
    "pseudo_beta": [NUM_RES, None],
    "pseudo_beta_mask": [NUM_RES],
    "pseudo_residue_feat": [],
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
    "symmetry_opers": [None, 3, 3],
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
    "template_frame_mask": [NUM_TEMPLATES, NUM_RES],
    "template_frame_tensor": [NUM_TEMPLATES, NUM_RES, None, None],
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
    "true_frame_tensor": [NUM_RES, None, None],
    "use_clamped_fape": [],
    "assembly_num_chains": [1],
    "asym_id": [NUM_RES],
    "sym_id": [NUM_RES],
    "entity_id": [NUM_RES],
    "num_sym": [NUM_RES],
    "asym_len": [None],
    "cluster_bias_mask": [NUM_MSA_SEQ],
}


@dataclass
class MaskedMSA:
    profile_prob: float = 0.1
    same_prob: float = 0.1
    uniform_prob: float = 0.1


@dataclass
class BlockDeleteMSA:
    msa_fraction_per_block: float = 0.3
    randomize_num_blocks: bool = False
    num_blocks: int = 5
    min_num_msa: int = 16


@dataclass
class RandomDeleteMSA:
    max_msa_entry: int = 33554432


@dataclass
class CommonData:
    masked_msa: MaskedMSA = MaskedMSA()
    max_recycling_iters: int = 3
    max_extra_msa: int = 1024
    msa_cluster_features: bool = True
    reduce_msa_clusters_by_max_templates: bool = True
    resample_msa_in_recycling: bool = True
    v2_feature: bool = False
    gumbel_sample: bool = False
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
        "msa_chains",
        "num_alignments",
        "seq_length",
        "between_segment_residues",
        "deletion_matrix",
        "num_recycling_iters",
        "crop_and_fix_size_seed",
    )
    use_templates: bool = True
    use_template_torsion_angles: bool = True
    is_multimer: bool = False
    recycling_features: Tuple[str, ...] = (
        "msa_chains",
        "msa_mask",
        "msa_row_mask",
        "bert_mask",
        "true_msa",
        "msa_feat",
        "extra_msa_deletion_value",
        "extra_msa_has_deletion",
        "extra_msa",
        "extra_msa_mask",
        "extra_msa_row_mask",
        "is_distillation",
    )
    multimer_features: Tuple[str, ...] = (
        "assembly_num_chains",
        "asym_id",
        "sym_id",
        "num_sym",
        "entity_id",
        "asym_len",
        "cluster_bias_mask",
    )
    random_delete_msa: RandomDeleteMSA = RandomDeleteMSA()
    block_delete_msa: BlockDeleteMSA = BlockDeleteMSA()


@dataclass
class PredictData:
    fixed_size: bool = True
    subsample_templates: bool = False
    masked_msa_replace_fraction: float = 0.15
    max_msa_clusters: int = 512
    max_templates: int = 4
    crop: bool = False
    crop_size: Optional[int] = None
    supervised: bool = False
    biased_msa_by_chain: bool = False
    share_mask: bool = False
    num_ensembles: int = 2
    random_delete_msa: bool = True
    block_delete_msa: bool = False


@dataclass
class DataConfig:
    common: CommonData = CommonData()
    predict: PredictData = PredictData()


@dataclass
class GlobalsConfig:
    block_size: Optional[int] = None
    d_pair: int = 128
    d_msa: int = 256
    d_template: int = 64
    d_extra_msa: int = 64
    d_single: int = 384
    chunk_size: int = 4
    eps: float = 1e-8
    inf: float = 3e4
    max_recycling_iters: int = 3
    alphafold_original_mode: bool = False


@dataclass
class InputEmbedderConfig:
    tf_dim: int = 22
    msa_dim: int = 49
    d_pair: float = 128
    d_msa: float = 256
    relpos_k: int = 32
    max_relative_chain: int = 2


@dataclass
class RecyclingEmbedderConfig:
    d_pair: int = 128
    d_msa: int = 256
    min_bin: float = 3.25
    max_bin: float = 20.75
    num_bins: int = 15
    inf: float = 3e4


@dataclass
class TemplateAngleEmbedder:
    d_in: int = 57
    d_out: int = 256


@dataclass
class TemplatePairEmbedder:
    d_in: int = 88
    d_out: int = 64
    v2_d_in: List[int] = field(default_factory=lambda: [39, 1, 22, 22, 1, 1, 1, 1])
    d_pair: int = 128
    v2_feature: bool = False


@dataclass
class TemplatePairStack:
    d_template: int = 64
    d_hid_tri_att: int = 16
    d_hid_tri_mul: int = 64
    num_blocks: int = 2
    num_heads: int = 4
    pair_transition_n: int = 2
    dropout_rate: float = 0.25
    inf: float = 3e4
    tri_attn_first: bool = True


@dataclass
class TemplatePointwiseAttention:
    d_template: int = 64
    d_pair: int = 128
    d_hid: int = 16
    num_heads: int = 4
    inf: float = 3e4
    enabled: bool = True


@dataclass
class Distogram:
    min_bin: float = 3.25
    max_bin: float = 50.75
    num_bins: int = 39


@dataclass
class TemplateConfig:
    distogram: Distogram = Distogram()
    template_angle_embedder: TemplateAngleEmbedder = TemplateAngleEmbedder()
    template_pair_embedder: TemplatePairEmbedder = TemplatePairEmbedder()
    template_pair_stack: TemplatePairStack = TemplatePairStack()
    template_pointwise_attention: TemplatePointwiseAttention = (
        TemplatePointwiseAttention()
    )
    inf: float = 3e4
    eps: float = 1e-6
    enabled: bool = True
    embed_angles: bool = True


@dataclass
class ExtraMsaEmbedder:
    d_in: int = 25
    d_out: int = 64


@dataclass
class ExtraMsaStack:
    d_msa: int = 64
    d_pair: int = 128
    d_hid_msa_att: int = 8
    d_hid_opm: int = 32
    d_hid_mul: int = 128
    d_hid_pair_att: int = 32
    num_heads_msa: int = 8
    num_heads_pair: int = 4
    num_blocks: int = 4
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    outer_product_mean_first: bool = False
    inf: float = 3e4
    eps: float = 1e-8


@dataclass
class ExtraMsaConfig:
    extra_msa_embedder: ExtraMsaEmbedder = ExtraMsaEmbedder()
    extra_msa_stack: ExtraMsaStack = ExtraMsaStack()
    enabled: bool = True


@dataclass
class EvoformerStack:
    d_msa: int = 256
    d_pair: int = 128
    d_hid_msa_att: int = 32
    d_hid_opm: int = 32
    d_hid_mul: int = 128
    d_hid_pair_att: int = 32
    d_single: int = 384
    num_heads_msa: int = 8
    num_heads_pair: int = 4
    num_blocks: int = 48
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    inf: float = 3e4
    eps: float = 1e-8
    outer_product_mean_first: bool = False


@dataclass
class StructureModule:
    d_single: int = 384
    d_pair: int = 128
    d_ipa: int = 16
    d_angle: int = 128
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    dropout_rate: float = 0.1
    num_blocks: int = 8
    no_transition_layers: int = 1
    num_resnet_blocks: int = 2
    num_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 3e4
    separate_kv: bool = False
    ipa_bias: bool = True


@dataclass
class HeadsPlddt:
    num_bins: int = 50
    d_in: int = 384
    d_hid: int = 128


@dataclass
class HeadsDistogram:
    d_pair: int = 128
    num_bins: int = 64
    disable_enhance_head: bool = False


@dataclass
class HeadsTm:
    d_pair: int = 128
    num_bins: int = 64
    enabled: bool = False


@dataclass
class HeadsMaskedMsa:
    d_msa: int = 256
    d_out: int = 23
    disable_enhance_head: bool = False


@dataclass
class HeadsExperimentallyResolved:
    d_single: int = 384
    d_out: int = 37
    enabled: bool = False
    disable_enhance_head: bool = False


@dataclass
class HeadsPAE:
    d_pair: int = 128
    num_bins: int = 64
    enabled: bool = False
    iptm_weight: float = 0.8
    disable_enhance_head: bool = False


@dataclass
class HeadsConfig:
    plddt: HeadsPlddt = HeadsPlddt()
    distogram: HeadsDistogram = HeadsDistogram()
    masked_msa: HeadsMaskedMsa = HeadsMaskedMsa()
    experimentally_resolved: HeadsExperimentallyResolved = HeadsExperimentallyResolved()
    pae: HeadsPAE = HeadsPAE()


@dataclass
class ModelConfig:
    input_embedder: InputEmbedderConfig = InputEmbedderConfig()
    recycling_embedder: RecyclingEmbedderConfig = RecyclingEmbedderConfig()
    template: TemplateConfig = TemplateConfig()
    extra_msa: ExtraMsaConfig = ExtraMsaConfig()
    evoformer_stack: EvoformerStack = EvoformerStack()
    structure_module: StructureModule = StructureModule()
    heads: HeadsConfig = HeadsConfig()
    is_multimer: bool = False


@dataclass
class DistogramLoss:
    min_bin: float = 2.3125
    max_bin: float = 21.6875
    num_bins: int = 64
    eps: float = 1e8
    weight: float = 0.3


@dataclass
class ExperimentallyResolvedLoss:
    eps: float = 1e-8
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    weight: float = 0.0


@dataclass
class PAELoss:
    max_bin: int = 31
    num_bins: int = 64
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    eps: float = 1e-8
    weight: float = 0.0


@dataclass
class BackboneFAPELoss:
    clamp_distance: float = 10.0
    clamp_distance_between_chains: float = 30.0
    loss_unit_distance: float = 10.0
    loss_unit_distance_between_chains: float = 20.0
    weight: float = 0.5
    eps: float = 1e-4


@dataclass
class SidechainFAPELoss:
    clamp_distance: float = 10.0
    length_scale: float = 10.0
    weight: float = 0.5
    eps: float = 1e-4


@dataclass
class FAPELoss:
    backbone: BackboneFAPELoss = BackboneFAPELoss()
    sidechain: SidechainFAPELoss = SidechainFAPELoss()
    weight: float = 1.0


@dataclass
class PLDDTLoss:
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    cutoff: float = 15.0
    num_bins: int = 50
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
    bond_angle_loss_weight: float = 0.3


@dataclass
class TMLoss:
    max_bin: int = 31
    num_bins: int = 64
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    eps: float = 1e-8
    weight: float = 0.0
    enabled: bool = False


@dataclass
class ReprNormLoss:
    weight: float = 0.01
    tolerance: float = 1.0


@dataclass
class ChainCentreMassLoss:
    weight: float = 0.0
    eps: float = 1e-8


@dataclass
class LossConfig:
    distogram: DistogramLoss = DistogramLoss()
    experimentally_resolved: ExperimentallyResolvedLoss = ExperimentallyResolvedLoss()
    fape: FAPELoss = FAPELoss()
    plddt: PLDDTLoss = PLDDTLoss()
    masked_msa: MaskedMSALoss = MaskedMSALoss()
    supervised_chi: SupervisedChiLoss = SupervisedChiLoss()
    violation: ViolationLoss = ViolationLoss()
    pae: PAELoss = PAELoss()
    repr_norm: ReprNormLoss = ReprNormLoss()
    chain_centre_mass: ChainCentreMassLoss = ChainCentreMassLoss()


@dataclass
class UniFoldConfig:
    data: DataConfig = DataConfig()
    globals: GlobalsConfig = GlobalsConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()


@dataclass
class Model2FT(UniFoldConfig):
    data: DataConfig = DataConfig(
        common=CommonData(max_extra_msa=1024), predict=PredictData(max_msa_clusters=512)
    )


@dataclass
class Model1AF2(UniFoldConfig):
    data: DataConfig = DataConfig(
        common=CommonData(max_extra_msa=5120), predict=PredictData(max_msa_clusters=512)
    )
    globals: GlobalsConfig = GlobalsConfig(alphafold_original_mode=True)
    model: ModelConfig = ModelConfig(
        heads=HeadsConfig(
            experimentally_resolved=HeadsExperimentallyResolved(enabled=True)
        )
    )
    loss: LossConfig = LossConfig(
        violation=ViolationLoss(weight=0.02),
        repr_norm=ReprNormLoss(weight=0),
        experimentally_resolved=ExperimentallyResolvedLoss(weight=0.01),
    )


@dataclass
class Model2AF2(Model1AF2):
    data: DataConfig = DataConfig(
        common=CommonData(max_extra_msa=1024),
        predict=PredictData(max_msa_clusters=512),
    )


@dataclass
class Model3AF2(Model1AF2):
    data: DataConfig = DataConfig(
        common=CommonData(use_templates=False, use_template_torsion_angles=False),
    )
    model: ModelConfig = ModelConfig(
        template=TemplateConfig(embed_angles=False, enabled=False),
        heads=HeadsConfig(
            experimentally_resolved=HeadsExperimentallyResolved(enabled=True)
        ),
    )


@dataclass
class Model5AF2(Model3AF2):
    data: DataConfig = DataConfig(
        common=CommonData(
            max_extra_msa=1024, use_templates=False, use_template_torsion_angles=False
        ),
        predict=PredictData(max_msa_clusters=512),
    )


@dataclass
class MultimerFT(UniFoldConfig):
    data: DataConfig = DataConfig(
        common=CommonData(
            is_multimer=True, max_extra_msa=1024, v2_feature=True, gumbel_sample=True
        ),
        predict=PredictData(max_msa_clusters=512),
    )
    model: ModelConfig = ModelConfig(
        heads=HeadsConfig(
            pae=HeadsPAE(enabled=True, disable_enhance_head=True),
            masked_msa=HeadsMaskedMsa(d_out=22),
        ),
        input_embedder=InputEmbedderConfig(tf_dim=21),
        structure_module=StructureModule(
            separate_kv=True, ipa_bias=False, trans_scale_factor=20
        ),
        template=TemplateConfig(
            template_angle_embedder=TemplateAngleEmbedder(d_in=34),
            template_pair_stack=TemplatePairStack(tri_attn_first=False),
            template_pair_embedder=TemplatePairEmbedder(v2_feature=True),
            template_pointwise_attention=TemplatePointwiseAttention(enabled=False),
        ),
        is_multimer=True,
    )
    loss: LossConfig = LossConfig(
        pae=PAELoss(weight=0.1),
        violation=ViolationLoss(weight=0.5),
        chain_centre_mass=ChainCentreMassLoss(weight=1.0),
    )


@dataclass
class MultimerAF2V3(UniFoldConfig):
    data: DataConfig = DataConfig(
        common=CommonData(
            is_multimer=True, max_extra_msa=2048, v2_feature=True, gumbel_sample=True
        ),
        predict=PredictData(max_msa_clusters=512),
    )
    globals: GlobalsConfig = GlobalsConfig(alphafold_original_mode=True)
    model: ModelConfig = ModelConfig(
        evoformer_stack=EvoformerStack(outer_product_mean_first=True),
        extra_msa=ExtraMsaConfig(
            extra_msa_stack=ExtraMsaStack(outer_product_mean_first=True)
        ),
        heads=HeadsConfig(
            pae=HeadsPAE(enabled=True),
            masked_msa=HeadsMaskedMsa(d_out=22),
            experimentally_resolved=HeadsExperimentallyResolved(enabled=True),
        ),
        input_embedder=InputEmbedderConfig(tf_dim=21),
        structure_module=StructureModule(
            separate_kv=True, ipa_bias=False, trans_scale_factor=20
        ),
        template=TemplateConfig(
            template_angle_embedder=TemplateAngleEmbedder(d_in=34),
            template_pair_stack=TemplatePairStack(tri_attn_first=False),
            template_pair_embedder=TemplatePairEmbedder(v2_feature=True),
            template_pointwise_attention=TemplatePointwiseAttention(enabled=False),
        ),
        is_multimer=True,
    )
    loss: LossConfig = LossConfig(
        pae=PAELoss(weight=0.1),
        violation=ViolationLoss(weight=0.5),
        chain_centre_mass=ChainCentreMassLoss(weight=1.0),
        repr_norm=ReprNormLoss(weight=0),
        experimentally_resolved=ExperimentallyResolvedLoss(weight=0.01),
    )


@dataclass
class MultimerAF2Model45V3(MultimerAF2V3):
    data: DataConfig = DataConfig(
        common=CommonData(
            is_multimer=True, max_extra_msa=1152, v2_feature=True, gumbel_sample=True
        ),
        predict=PredictData(max_msa_clusters=512),
    )


@dataclass
class SymmetryInputEmbedderConfig(InputEmbedderConfig):
    pr_dim: int = 48
    tf_dim: int = 21


@dataclass
class PseudoResidueEmbedder:
    d_in: int = 8
    d_hidden: int = 48
    d_out: int = 48
    num_blocks: int = 4


@dataclass
class SymmetryModelConfig(ModelConfig):
    heads: HeadsConfig = HeadsConfig(
        pae=HeadsPAE(enabled=True, disable_enhance_head=True),
        masked_msa=HeadsMaskedMsa(d_out=22),
        experimentally_resolved=HeadsExperimentallyResolved(enabled=True),
    )
    input_embedder: SymmetryInputEmbedderConfig = SymmetryInputEmbedderConfig()
    structure_module: StructureModule = StructureModule(
        separate_kv=True, ipa_bias=False, trans_scale_factor=20
    )
    template: TemplateConfig = TemplateConfig(
        template_angle_embedder=TemplateAngleEmbedder(d_in=34),
        template_pair_stack=TemplatePairStack(tri_attn_first=False),
        template_pair_embedder=TemplatePairEmbedder(v2_feature=True),
        template_pointwise_attention=TemplatePointwiseAttention(enabled=False),
    )
    pseudo_residue_embedder: PseudoResidueEmbedder = PseudoResidueEmbedder()
    is_multimer: bool = True


@dataclass
class UniFoldSymmetry(MultimerFT):
    data: DataConfig = DataConfig(
        common=CommonData(
            is_multimer=True, max_extra_msa=1024, v2_feature=True, gumbel_sample=True
        ),
        predict=PredictData(max_msa_clusters=256),
    )
    model: SymmetryModelConfig = SymmetryModelConfig()
    loss: LossConfig = LossConfig(
        pae=PAELoss(weight=0.0),
        violation=ViolationLoss(weight=0.5),
        chain_centre_mass=ChainCentreMassLoss(weight=1.0),
        experimentally_resolved=ExperimentallyResolvedLoss(weight=0.0),
    )


def make_data_config_dataclass(
    config: DataConfig,
    num_res: int,
    use_templates: bool = False,
    is_multimer: bool = False,
) -> Tuple[DataConfig, List[str]]:
    """Make a data config dataclass with the given number of residues.

    Args:
        config: The data config dataclass.
        num_res: The number of residues.
        use_templates: Whether to use templates.
        is_multimer: Whether the model is a multimer.

    Returns:
        The data config dataclass and the list of feature names.
    """
    cfg = copy.deepcopy(config)
    mode_cfg = cfg.predict
    common_cfg = cfg.common
    if mode_cfg.crop_size is None:
        mode_cfg.crop_size = num_res

    feature_names = common_cfg.unsupervised_features + common_cfg.recycling_features
    if use_templates:
        feature_names += common_cfg.template_features
    if is_multimer:
        feature_names += common_cfg.multimer_features

    return cfg, feature_names
