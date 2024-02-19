import copy
from dataclasses import dataclass, field, fields, is_dataclass, _MISSING_TYPE
from typing import Any, List, Optional, Tuple

import ml_collections as mlc
import numpy as np

N_RES = "number of residues"
N_MSA = "number of MSA sequences"
N_EXTRA_MSA = "number of extra MSA sequences"
N_TPL = "number of templates"


d_pair = mlc.FieldReference(128, field_type=int)  # c_z
d_msa = mlc.FieldReference(256, field_type=int)  # c_m
d_template = mlc.FieldReference(64, field_type=int)  # c_t
d_extra_msa = mlc.FieldReference(64, field_type=int)  # c_e
d_single = mlc.FieldReference(384, field_type=int)  # c_s
max_recycling_iters = mlc.FieldReference(3, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)
inf = mlc.FieldReference(3e4, field_type=float)
use_templates = mlc.FieldReference(True, field_type=bool)
is_multimer = mlc.FieldReference(False, field_type=bool)


def base_config():
    return mlc.ConfigDict(
        {
            "data": {
                "common": {
                    "features": {
                        "aatype": [N_RES],
                        "all_atom_mask": [N_RES, None],
                        "all_atom_positions": [N_RES, None, None],
                        "alt_chi_angles": [N_RES, None],
                        "atom14_alt_gt_exists": [N_RES, None],
                        "atom14_alt_gt_positions": [N_RES, None, None],
                        "atom14_atom_exists": [N_RES, None],
                        "atom14_atom_is_ambiguous": [N_RES, None],
                        "atom14_gt_exists": [N_RES, None],
                        "atom14_gt_positions": [N_RES, None, None],
                        "atom37_atom_exists": [N_RES, None],
                        "frame_mask": [N_RES],
                        "true_frame_tensor": [N_RES, None, None],
                        "bert_mask": [N_MSA, N_RES],
                        "chi_angles_sin_cos": [N_RES, None, None],
                        "chi_mask": [N_RES, None],
                        "extra_msa_deletion_value": [N_EXTRA_MSA, N_RES],
                        "extra_msa_has_deletion": [N_EXTRA_MSA, N_RES],
                        "extra_msa": [N_EXTRA_MSA, N_RES],
                        "extra_msa_mask": [N_EXTRA_MSA, N_RES],
                        "extra_msa_row_mask": [N_EXTRA_MSA],
                        "is_distillation": [],
                        "msa_feat": [N_MSA, N_RES, None],
                        "msa_mask": [N_MSA, N_RES],
                        "msa_chains": [N_MSA, None],
                        "msa_row_mask": [N_MSA],
                        "num_recycling_iters": [],
                        "pseudo_beta": [N_RES, None],
                        "pseudo_beta_mask": [N_RES],
                        "residue_index": [N_RES],
                        "residx_atom14_to_atom37": [N_RES, None],
                        "residx_atom37_to_atom14": [N_RES, None],
                        "resolution": [],
                        "rigidgroups_alt_gt_frames": [N_RES, None, None, None],
                        "rigidgroups_group_exists": [N_RES, None],
                        "rigidgroups_group_is_ambiguous": [N_RES, None],
                        "rigidgroups_gt_exists": [N_RES, None],
                        "rigidgroups_gt_frames": [N_RES, None, None, None],
                        "seq_length": [],
                        "seq_mask": [N_RES],
                        "target_feat": [N_RES, None],
                        "template_aatype": [N_TPL, N_RES],
                        "template_all_atom_mask": [N_TPL, N_RES, None],
                        "template_all_atom_positions": [N_TPL, N_RES, None, None],
                        "template_alt_torsion_angles_sin_cos": [
                            N_TPL,
                            N_RES,
                            None,
                            None,
                        ],
                        "template_frame_mask": [N_TPL, N_RES],
                        "template_frame_tensor": [N_TPL, N_RES, None, None],
                        "template_mask": [N_TPL],
                        "template_pseudo_beta": [N_TPL, N_RES, None],
                        "template_pseudo_beta_mask": [N_TPL, N_RES],
                        "template_sum_probs": [N_TPL, None],
                        "template_torsion_angles_mask": [N_TPL, N_RES, None],
                        "template_torsion_angles_sin_cos": [N_TPL, N_RES, None, None],
                        "true_msa": [N_MSA, N_RES],
                        "use_clamped_fape": [],
                        "assembly_num_chains": [1],
                        "asym_id": [N_RES],
                        "sym_id": [N_RES],
                        "entity_id": [N_RES],
                        "num_sym": [N_RES],
                        "asym_len": [None],
                        "cluster_bias_mask": [N_MSA],
                    },
                    "masked_msa": {
                        "profile_prob": 0.1,
                        "same_prob": 0.1,
                        "uniform_prob": 0.1,
                    },
                    "block_delete_msa": {
                        "msa_fraction_per_block": 0.3,
                        "randomize_num_blocks": False,
                        "num_blocks": 5,
                        "min_num_msa": 16,
                    },
                    "random_delete_msa": {
                        "max_msa_entry": 1 << 25,  # := 33554432
                    },
                    "v2_feature": False,
                    "gumbel_sample": False,
                    "max_extra_msa": 1024,
                    "msa_cluster_features": True,
                    "reduce_msa_clusters_by_max_templates": True,
                    "resample_msa_in_recycling": True,
                    "template_features": [
                        "template_all_atom_positions",
                        "template_sum_probs",
                        "template_aatype",
                        "template_all_atom_mask",
                    ],
                    "unsupervised_features": [
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
                    ],
                    "recycling_features": [
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
                    ],
                    "multimer_features": [
                        "assembly_num_chains",
                        "asym_id",
                        "sym_id",
                        "num_sym",
                        "entity_id",
                        "asym_len",
                        "cluster_bias_mask",
                    ],
                    "use_templates": use_templates,
                    "is_multimer": is_multimer,
                    "use_template_torsion_angles": use_templates,
                    "max_recycling_iters": max_recycling_iters,
                },
                "supervised": {
                    "use_clamped_fape_prob": 1.0,
                    "supervised_features": [
                        "all_atom_mask",
                        "all_atom_positions",
                        "resolution",
                        "use_clamped_fape",
                        "is_distillation",
                    ],
                },
                "predict": {
                    "fixed_size": True,
                    "subsample_templates": False,
                    "block_delete_msa": False,
                    "random_delete_msa": True,
                    "masked_msa_replace_fraction": 0.15,
                    "max_msa_clusters": 128,
                    "max_templates": 4,
                    "num_ensembles": 2,
                    "crop": False,
                    "crop_size": None,
                    "supervised": False,
                    "biased_msa_by_chain": False,
                    "share_mask": False,
                },
                "eval": {
                    "fixed_size": True,
                    "subsample_templates": False,
                    "block_delete_msa": False,
                    "random_delete_msa": True,
                    "masked_msa_replace_fraction": 0.15,
                    "max_msa_clusters": 128,
                    "max_templates": 4,
                    "num_ensembles": 1,
                    "crop": False,
                    "crop_size": None,
                    "spatial_crop_prob": 0.5,
                    "ca_ca_threshold": 10.0,
                    "supervised": True,
                    "biased_msa_by_chain": False,
                    "share_mask": False,
                },
                "train": {
                    "fixed_size": True,
                    "subsample_templates": True,
                    "block_delete_msa": True,
                    "random_delete_msa": True,
                    "masked_msa_replace_fraction": 0.15,
                    "max_msa_clusters": 128,
                    "max_templates": 4,
                    "num_ensembles": 1,
                    "crop": True,
                    "crop_size": 256,
                    "spatial_crop_prob": 0.5,
                    "ca_ca_threshold": 10.0,
                    "supervised": True,
                    "use_clamped_fape_prob": 1.0,
                    "max_distillation_msa_clusters": 1000,
                    "biased_msa_by_chain": True,
                    "share_mask": True,
                },
            },
            "globals": {
                "chunk_size": chunk_size,
                "block_size": None,
                "d_pair": d_pair,
                "d_msa": d_msa,
                "d_template": d_template,
                "d_extra_msa": d_extra_msa,
                "d_single": d_single,
                "eps": eps,
                "inf": inf,
                "max_recycling_iters": max_recycling_iters,
                "alphafold_original_mode": False,
            },
            "model": {
                "is_multimer": is_multimer,
                "input_embedder": {
                    "tf_dim": 22,
                    "msa_dim": 49,
                    "d_pair": d_pair,
                    "d_msa": d_msa,
                    "relpos_k": 32,
                    "max_relative_chain": 2,
                },
                "recycling_embedder": {
                    "d_pair": d_pair,
                    "d_msa": d_msa,
                    "min_bin": 3.25,
                    "max_bin": 20.75,
                    "num_bins": 15,
                    "inf": 1e8,
                },
                "template": {
                    "distogram": {
                        "min_bin": 3.25,
                        "max_bin": 50.75,
                        "num_bins": 39,
                    },
                    "template_angle_embedder": {
                        "d_in": 57,
                        "d_out": d_msa,
                    },
                    "template_pair_embedder": {
                        "d_in": 88,
                        "v2_d_in": [39, 1, 22, 22, 1, 1, 1, 1],
                        "d_pair": d_pair,
                        "d_out": d_template,
                        "v2_feature": False,
                    },
                    "template_pair_stack": {
                        "d_template": d_template,
                        "d_hid_tri_att": 16,
                        "d_hid_tri_mul": 64,
                        "num_blocks": 2,
                        "num_heads": 4,
                        "pair_transition_n": 2,
                        "dropout_rate": 0.25,
                        "inf": 1e9,
                        "tri_attn_first": True,
                    },
                    "template_pointwise_attention": {
                        "enabled": True,
                        "d_template": d_template,
                        "d_pair": d_pair,
                        "d_hid": 16,
                        "num_heads": 4,
                        "inf": 1e5,
                    },
                    "inf": 1e5,
                    "eps": 1e-6,
                    "enabled": use_templates,
                    "embed_angles": use_templates,
                },
                "extra_msa": {
                    "extra_msa_embedder": {
                        "d_in": 25,
                        "d_out": d_extra_msa,
                    },
                    "extra_msa_stack": {
                        "d_msa": d_extra_msa,
                        "d_pair": d_pair,
                        "d_hid_msa_att": 8,
                        "d_hid_opm": 32,
                        "d_hid_mul": 128,
                        "d_hid_pair_att": 32,
                        "num_heads_msa": 8,
                        "num_heads_pair": 4,
                        "num_blocks": 4,
                        "transition_n": 4,
                        "msa_dropout": 0.15,
                        "pair_dropout": 0.25,
                        "inf": 1e9,
                        "eps": 1e-10,
                        "outer_product_mean_first": False,
                    },
                    "enabled": True,
                },
                "evoformer_stack": {
                    "d_msa": d_msa,
                    "d_pair": d_pair,
                    "d_hid_msa_att": 32,
                    "d_hid_opm": 32,
                    "d_hid_mul": 128,
                    "d_hid_pair_att": 32,
                    "d_single": d_single,
                    "num_heads_msa": 8,
                    "num_heads_pair": 4,
                    "num_blocks": 48,
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "inf": 1e9,
                    "eps": 1e-10,
                    "outer_product_mean_first": False,
                },
                "structure_module": {
                    "d_single": d_single,
                    "d_pair": d_pair,
                    "d_ipa": 16,
                    "d_angle": 128,
                    "num_heads_ipa": 12,
                    "num_qk_points": 4,
                    "num_v_points": 8,
                    "dropout_rate": 0.1,
                    "num_blocks": 8,
                    "no_transition_layers": 1,
                    "num_resnet_blocks": 2,
                    "num_angles": 7,
                    "trans_scale_factor": 10,
                    "epsilon": 1e-12,
                    "inf": 1e5,
                    "separate_kv": False,
                    "ipa_bias": True,
                },
                "heads": {
                    "plddt": {
                        "num_bins": 50,
                        "d_in": d_single,
                        "d_hid": 128,
                    },
                    "distogram": {
                        "d_pair": d_pair,
                        "num_bins": aux_distogram_bins,
                        "disable_enhance_head": False,
                    },
                    "pae": {
                        "d_pair": d_pair,
                        "num_bins": aux_distogram_bins,
                        "enabled": False,
                        "iptm_weight": 0.8,
                        "disable_enhance_head": False,
                    },
                    "masked_msa": {
                        "d_msa": d_msa,
                        "d_out": 23,
                        "disable_enhance_head": False,
                    },
                    "experimentally_resolved": {
                        "d_single": d_single,
                        "d_out": 37,
                        "enabled": False,
                        "disable_enhance_head": False,
                    },
                },
            },
            "loss": {
                "distogram": {
                    "min_bin": 2.3125,
                    "max_bin": 21.6875,
                    "num_bins": 64,
                    "eps": 1e-6,
                    "weight": 0.3,
                },
                "experimentally_resolved": {
                    "eps": 1e-8,
                    "min_resolution": 0.1,
                    "max_resolution": 3.0,
                    "weight": 0.0,
                },
                "fape": {
                    "backbone": {
                        "clamp_distance": 10.0,
                        "clamp_distance_between_chains": 30.0,
                        "loss_unit_distance": 10.0,
                        "loss_unit_distance_between_chains": 20.0,
                        "weight": 0.5,
                        "eps": 1e-4,
                    },
                    "sidechain": {
                        "clamp_distance": 10.0,
                        "length_scale": 10.0,
                        "weight": 0.5,
                        "eps": 1e-4,
                    },
                    "weight": 1.0,
                },
                "plddt": {
                    "min_resolution": 0.1,
                    "max_resolution": 3.0,
                    "cutoff": 15.0,
                    "num_bins": 50,
                    "eps": 1e-10,
                    "weight": 0.01,
                },
                "masked_msa": {
                    "eps": 1e-8,
                    "weight": 2.0,
                },
                "supervised_chi": {
                    "chi_weight": 0.5,
                    "angle_norm_weight": 0.01,
                    "eps": 1e-6,
                    "weight": 1.0,
                },
                "violation": {
                    "violation_tolerance_factor": 12.0,
                    "clash_overlap_tolerance": 1.5,
                    "bond_angle_loss_weight": 0.3,
                    "eps": 1e-6,
                    "weight": 0.0,
                },
                "pae": {
                    "max_bin": 31,
                    "num_bins": 64,
                    "min_resolution": 0.1,
                    "max_resolution": 3.0,
                    "eps": 1e-8,
                    "weight": 0.0,
                },
                "repr_norm": {
                    "weight": 0.01,
                    "tolerance": 1.0,
                },
                "chain_centre_mass": {
                    "weight": 0.0,
                    "eps": 1e-8,
                },
            },
        }
    )


def recursive_set(c: mlc.ConfigDict, key: str, value: Any, ignore: str = None):
    with c.unlocked():
        for k, v in c.items():
            if ignore is not None and k == ignore:
                continue
            if isinstance(v, mlc.ConfigDict):
                recursive_set(v, key, value)
            elif k == key:
                c[k] = value


def model_config(name, train=False):
    c = copy.deepcopy(base_config())

    def multimer(c):
        recursive_set(c, "is_multimer", True)
        recursive_set(c, "max_extra_msa", 1152)
        recursive_set(c, "max_msa_clusters", 128)
        recursive_set(c, "v2_feature", True)
        recursive_set(c, "gumbel_sample", True)
        c.model.template.template_angle_embedder.d_in = 34
        c.model.template.template_pair_stack.tri_attn_first = False
        c.model.template.template_pointwise_attention.enabled = False
        c.model.heads.pae.enabled = True
        # we forget to enable it in our training, so disable it here
        c.model.heads.pae.disable_enhance_head = True
        c.model.heads.masked_msa.d_out = 22
        c.model.structure_module.separate_kv = True
        c.model.structure_module.ipa_bias = False
        c.model.structure_module.trans_scale_factor = 20
        c.loss.pae.weight = 0.1
        c.model.input_embedder.tf_dim = 21
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.chain_centre_mass.weight = 1.0
        return c

    if name == "model_1_af2":
        recursive_set(c, "max_extra_msa", 5120)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.repr_norm.weight = 0
        c.model.heads.experimentally_resolved.enabled = True
        c.loss.experimentally_resolved.weight = 0.01
        c.globals.alphafold_original_mode = True
    elif name == "model_2_ft":
        recursive_set(c, "max_extra_msa", 1024)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
    elif name == "model_2_af2":
        recursive_set(c, "max_extra_msa", 1024)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.repr_norm.weight = 0
        c.model.heads.experimentally_resolved.enabled = True
        c.loss.experimentally_resolved.weight = 0.01
        c.globals.alphafold_original_mode = True
    elif name == "model_3_af2" or name == "model_4_af2":
        recursive_set(c, "max_extra_msa", 5120)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.repr_norm.weight = 0
        c.model.heads.experimentally_resolved.enabled = True
        c.loss.experimentally_resolved.weight = 0.01
        c.globals.alphafold_original_mode = True
        c.model.template.enabled = False
        c.model.template.embed_angles = False
        recursive_set(c, "use_templates", False)
        recursive_set(c, "use_template_torsion_angles", False)
    elif name == "model_5_af2":
        recursive_set(c, "max_extra_msa", 1024)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.repr_norm.weight = 0
        c.model.heads.experimentally_resolved.enabled = True
        c.loss.experimentally_resolved.weight = 0.01
        c.globals.alphafold_original_mode = True
        c.model.template.enabled = False
        c.model.template.embed_angles = False
        recursive_set(c, "use_templates", False)
        recursive_set(c, "use_template_torsion_angles", False)
    elif name == "multimer_ft":
        c = multimer(c)
        recursive_set(c, "max_extra_msa", 1152)
        recursive_set(c, "max_msa_clusters", 256)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.5
    elif name == "multimer_af2_v3":
        recursive_set(c, "max_extra_msa", 2048)
        recursive_set(c, "max_msa_clusters", 512)
        recursive_set(c, "is_multimer", True)
        recursive_set(c, "v2_feature", True)
        recursive_set(c, "gumbel_sample", True)
        c.model.template.template_angle_embedder.d_in = 34
        c.model.template.template_pair_stack.tri_attn_first = False
        c.model.template.template_pointwise_attention.enabled = False
        c.model.heads.pae.enabled = True
        c.model.heads.experimentally_resolved.enabled = True
        c.model.heads.masked_msa.d_out = 22
        c.model.structure_module.separate_kv = True
        c.model.structure_module.ipa_bias = False
        c.model.structure_module.trans_scale_factor = 20
        c.loss.pae.weight = 0.1
        c.loss.violation.weight = 0.5
        c.loss.experimentally_resolved.weight = 0.01
        c.model.input_embedder.tf_dim = 21
        c.globals.alphafold_original_mode = True
        c.data.train.crop_size = 384
        c.loss.repr_norm.weight = 0
        c.loss.chain_centre_mass.weight = 1.0
        recursive_set(c, "outer_product_mean_first", True)
    elif name == "multimer_af2_model45_v3":
        recursive_set(c, "max_extra_msa", 1152)
        recursive_set(c, "max_msa_clusters", 512)
        recursive_set(c, "is_multimer", True)
        recursive_set(c, "v2_feature", True)
        recursive_set(c, "gumbel_sample", True)
        c.model.template.template_angle_embedder.d_in = 34
        c.model.template.template_pair_stack.tri_attn_first = False
        c.model.template.template_pointwise_attention.enabled = False
        c.model.heads.pae.enabled = True
        c.model.heads.experimentally_resolved.enabled = True
        c.model.heads.masked_msa.d_out = 22
        c.model.structure_module.separate_kv = True
        c.model.structure_module.ipa_bias = False
        c.model.structure_module.trans_scale_factor = 20
        c.loss.pae.weight = 0.1
        c.loss.violation.weight = 0.5
        c.loss.experimentally_resolved.weight = 0.01
        c.model.input_embedder.tf_dim = 21
        c.globals.alphafold_original_mode = True
        c.data.train.crop_size = 384
        c.loss.repr_norm.weight = 0
        c.loss.chain_centre_mass.weight = 1.0
        recursive_set(c, "outer_product_mean_first", True)
    else:
        raise ValueError(f"invalid --model-name: {name}.")
    if train:
        c.globals.chunk_size = None

    recursive_set(c, "inf", 3e4)
    recursive_set(c, "eps", 1e-5, "loss")
    return c


def make_data_config(
    config: mlc.ConfigDict,
    mode: str,
    num_res: int,
    use_templates: bool = False,
    is_multimer: bool = False,
) -> Tuple[mlc.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res
    feature_names = cfg.common.unsupervised_features + cfg.common.recycling_features
    if use_templates:
        feature_names += cfg.common.template_features
    if is_multimer:
        feature_names += cfg.common.multimer_features
    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names


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
    "num_recycling_iters": [],
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
class Features:
    aatype: Optional[np.ndarray] = None
    all_atom_mask: Optional[np.ndarray] = None
    all_atom_positions: Optional[np.ndarray] = None
    alt_chi_angles: Optional[np.ndarray] = None
    assembly_num_chains: Optional[np.ndarray] = None
    asym_id: Optional[np.ndarray] = None
    asym_len: Optional[np.ndarray] = None
    atom14_alt_gt_exists: Optional[np.ndarray] = None
    atom14_alt_gt_positions: Optional[np.ndarray] = None
    atom14_atom_exists: Optional[np.ndarray] = None
    atom14_atom_is_ambiguous: Optional[np.ndarray] = None
    atom14_gt_exists: Optional[np.ndarray] = None
    atom14_gt_positions: Optional[np.ndarray] = None
    atom37_atom_exists: Optional[np.ndarray] = None
    bert_mask: Optional[np.ndarray] = None
    chi_angles_sin_cos: Optional[np.ndarray] = None
    chi_mask: Optional[np.ndarray] = None
    cluster_bias_mask: Optional[np.ndarray] = None
    entity_id: Optional[np.ndarray] = None
    extra_msa_deletion_value: Optional[np.ndarray] = None
    extra_msa_has_deletion: Optional[np.ndarray] = None
    extra_msa: Optional[np.ndarray] = None
    extra_msa_mask: Optional[np.ndarray] = None
    extra_msa_row_mask: Optional[np.ndarray] = None
    frame_mask: Optional[np.ndarray] = None
    is_distillation: Optional[np.ndarray] = None
    msa_chains: Optional[np.ndarray] = None
    msa_feat: Optional[np.ndarray] = None
    msa_mask: Optional[np.ndarray] = None
    msa_row_mask: Optional[np.ndarray] = None
    num_recycling_iters: Optional[np.ndarray] = None
    num_sym: Optional[np.ndarray] = None
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
    sym_id: Optional[np.ndarray] = None
    target_feat: Optional[np.ndarray] = None
    template_aatype: Optional[np.ndarray] = None
    template_all_atom_mask: Optional[np.ndarray] = None
    template_all_atom_positions: Optional[np.ndarray] = None
    template_alt_torsion_angles_sin_cos: Optional[np.ndarray] = None
    template_frame_tensor: Optional[np.ndarray] = None
    template_frame_mask: Optional[np.ndarray] = None
    template_mask: Optional[np.ndarray] = None
    template_pseudo_beta: Optional[np.ndarray] = None
    template_pseudo_beta_mask: Optional[np.ndarray] = None
    template_sum_probs: Optional[np.ndarray] = None
    template_torsion_angles_mask: Optional[np.ndarray] = None
    template_torsion_angles_sin_cos: Optional[np.ndarray] = None
    true_msa: Optional[np.ndarray] = None
    true_frame_tensor: Optional[np.ndarray] = None
    use_clamped_fape: Optional[np.ndarray] = None


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
class DataModule:
    use_small_bfd: bool = False


@dataclass
class DataConfig:
    common: CommonData = CommonData()
    predict: PredictData = PredictData()
    # Train and Eval and Supervised modes?


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
    inf: float = 1e9
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
    inf: float = 1e8



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
    inf: float = 1e9
    tri_attn_first: bool = True


@dataclass
class TemplatePointwiseAttention:
    d_template: int = 64
    d_pair: int = 128
    d_hid: int = 16
    num_heads: int = 4
    inf: float = 1e5
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
    inf: float = 1e9
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
    inf: float = 1e9
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
    inf: float = 1e9
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
    inf: float = 1e5
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
class EmaConfig:
    decay: float = 0.999
    warmup: int = 1000
    enabled: bool = True


def recursive_set_dataclass(
    data_cls: _MISSING_TYPE, key: str, value: Any, ignore: Optional[str] = None
):
    for k in fields(data_cls):
        if k.name == ignore:
            continue
        # Check if the value is a dataclass
        if is_dataclass(getattr(data_cls, k.name)):
            recursive_set_dataclass(getattr(data_cls, k.name), key, value)
        elif k.name == key:
            setattr(data_cls, key, value)


@dataclass
class UniFoldConfig:
    data: DataConfig = DataConfig()
    globals: GlobalsConfig = GlobalsConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()

    def __post_init__(self):
        recursive_set_dataclass(self, "inf", 3e4)
        recursive_set_dataclass(self, "eps", 1e-5, "loss")


@dataclass
class Model2FT(UniFoldConfig):
    data: DataConfig = DataConfig(
        common=CommonData(max_extra_msa=1024),
        predict=PredictData(max_msa_clusters=512)
    )


@dataclass
class Model1AF2(UniFoldConfig):
    data: DataConfig = DataConfig(
        common=CommonData(max_extra_msa=5120),
        predict=PredictData(max_msa_clusters=512)
    )
    globals: GlobalsConfig = GlobalsConfig(
        alphafold_original_mode=True
    )
    model: ModelConfig = ModelConfig(
        heads=HeadsConfig(
            experimentally_resolved=HeadsExperimentallyResolved(enabled=True)
        )
    )
    loss: LossConfig = LossConfig(
        violation=ViolationLoss(weight=0.02),
        repr_norm=ReprNormLoss(weight=0),
        experimentally_resolved=ExperimentallyResolvedLoss(weight=0.01)
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
        )
    )


@dataclass
class Model5AF2(Model3AF2):
    data: DataConfig = DataConfig(
        common=CommonData(max_extra_msa=1024, use_templates=False, use_template_torsion_angles=False),
        predict=PredictData(max_msa_clusters=512),
    )


@dataclass
class MultimerFT(UniFoldConfig):
    data: DataConfig = DataConfig(
        common=CommonData(is_multimer=True, max_extra_msa=1024, v2_feature=True, gumbel_sample=True),
        predict=PredictData(max_msa_clusters=512),
    )
    model: ModelConfig = ModelConfig(
        heads=HeadsConfig(
            pae=HeadsPAE(enabled=True, disable_enhance_head=True),
            masked_msa=HeadsMaskedMsa(d_out=22),
        ),
        input_embedder=InputEmbedderConfig(tf_dim=21),
        structure_module=StructureModule(separate_kv=True, ipa_bias=False, trans_scale_factor=20),
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
        common=CommonData(is_multimer=True, max_extra_msa=2048, v2_feature=True, gumbel_sample=True),
        predict=PredictData(max_msa_clusters=512),
    )
    globals: GlobalsConfig = GlobalsConfig(alphafold_original_mode=True)
    model: ModelConfig = ModelConfig(
        evoformer_stack=EvoformerStack(outer_product_mean_first=True),
        extra_msa=ExtraMsaConfig(extra_msa_stack=ExtraMsaStack(outer_product_mean_first=True)),
        heads=HeadsConfig(
            pae=HeadsPAE(enabled=True),
            masked_msa=HeadsMaskedMsa(d_out=22),
            experimentally_resolved=HeadsExperimentallyResolved(enabled=True),
        ),
        input_embedder=InputEmbedderConfig(tf_dim=21),
        structure_module=StructureModule(separate_kv=True, ipa_bias=False, trans_scale_factor=20),
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
        experimentally_resolved=ExperimentallyResolvedLoss(weight=0.01)
    )


@dataclass
class MultimerAF2Model45V3(MultimerAF2V3):
    data: DataConfig = DataConfig(
        common=CommonData(is_multimer=True, max_extra_msa=1152, v2_feature=True, gumbel_sample=True),
        predict=PredictData(max_msa_clusters=512),
    )


def make_data_config_dataclass(
    config: DataConfig,
    num_res: int,
    use_templates: bool = False,
    is_multimer: bool = False,
) -> Tuple[mlc.ConfigDict, List[str]]:
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