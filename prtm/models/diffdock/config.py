from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InferenceConfig:
    actual_steps: int = 19
    inf_sched_alpha: int = 1
    inf_sched_beta: int = 1
    inference_steps: int = 20
    initial_noise_std_proportion: float = 1.4601642460337794
    limit_failures: int = 5
    no_final_step_noise: bool = True
    no_model: bool = False
    no_random: bool = False
    no_random_pocket: bool = False
    ode: bool = False
    resample_rdkit: bool = False
    sigma_schedule: str = "expbeta"
    temp_psi_rot: float = 0.9022615585677628
    temp_psi_tor: float = 0.5946212391366862
    temp_psi_tr: float = 0.727287304570729
    temp_sampling_rot: float = 2.06391612594481
    temp_sampling_tor: float = 7.044261621607846
    temp_sampling_tr: float = 1.170050527854316
    temp_sigma_data_rot: float = 0.7464326999906034
    temp_sigma_data_tor: float = 0.6943254174849822
    temp_sigma_data_tr: float = 0.9299802531572672


@dataclass
class ScoreModelConfig:
    affinity_prediction: bool = False
    all_atoms: bool = False
    asyncronous_noise_schedule: bool = False
    atom_max_neighbors: int = 8
    atom_num_confidence_outputs: int = 1
    atom_radius: int = 5
    atom_confidence: bool = False
    backbone_loss_weight: int = 0
    batch_norm: bool = True
    batch_size: int = 12
    c_alpha_max_neighbors: int = 24
    center_max_distance: int = 30
    confidence_dropout: float = 0.0
    confidence_no_batchnorm: bool = False
    confidence_mode: bool = False
    confidence_weight: float = 0.33
    crop_beyond: int = 20
    cross_distance_embed_dim: int = 64
    cross_max_distance: int = 80
    dataset: str = "generalisation"
    depthwise_convolution: bool = False
    differentiate_convolutions: bool = True
    distance_embed_dim: int = 64
    dropout: float = 0.1
    dynamic_max_cross: bool = True
    embed_also_ligand: bool = True
    embedding_scale: int = 1000
    embedding_type: str = "sinusoidal"
    fixed_center_conv: bool = True
    include_miscellaneous_atoms: bool = False
    in_lig_edge_features: int = 4
    inf_pocket_cutoff: int = 5
    inf_pocket_knowledge: bool = False
    inf_sched_alpha: int = 1
    inf_sched_beta: int = 1
    inference_steps: int = 20
    knn_only_graph: bool = False
    lig_max_radius: int  = 5
    limit_complexes: int = 0
    lm_embedding_type: Optional[str] = "precomputed"
    max_radius: float = 5.0
    no_aminoacid_identities: bool = False
    no_torsion: bool = False
    norm_by_sigma: bool = False
    ns: int = 48
    num_confidence_outputs: int = 1
    num_conformers: int = 1
    num_conv_layers: int = 11
    num_dataloader_workers: int = 1
    num_inference_complexes: int = 500
    num_prot_emb_layers: int = 3
    nv: int = 10
    odd_parity: bool = False
    receptor_radius: float = 15.0
    rec_max_radius: int = 30
    reduce_pseudoscalars: bool = True
    remove_hs: bool = True
    rot_sigma_max: float = 1.55
    rot_sigma_min: float = 0.03
    scale_by_sigma: bool = True
    separate_noise_schedule: bool = False
    sh_lmax: int = 2
    sidechain_pred: bool = False
    sigma_embed_dim: int = 64
    sigma_schedule: str = "expbeta"
    smooth_edges: bool = False
    tor_sigma_max: float = 3.14
    tor_sigma_min: float = 0.0314
    tp_weights_layers: int = 2
    tr_sigma_max: float = 19.0
    tr_sigma_min: float = 0.1
    tr_weight: float = 0.33
    use_second_order_repr: bool = False
