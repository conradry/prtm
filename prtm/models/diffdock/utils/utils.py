import torch
from torch_geometric.nn.data_parallel import DataParallel
from torch_geometric.utils import subgraph

from prtm.models.diffdock.models.cg_model import CGModel
from prtm.models.diffdock.utils.diffusion_utils import get_timestep_embedding


def get_model(
    args,
    device,
    t_to_sigma,
    no_parallel=False,
):
    timestep_emb_func = get_timestep_embedding(
        embedding_type=(
            args.embedding_type if "embedding_type" in args else "sinusoidal"
        ),
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale if "embedding_type" in args else 10000,
    )

    model_class = CGModel
    model = model_class(
        t_to_sigma=t_to_sigma,
        device=device,
        no_torsion=args.no_torsion,
        timestep_emb_func=timestep_emb_func,
        num_conv_layers=args.num_conv_layers,
        lig_max_radius=args.max_radius,
        scale_by_sigma=args.scale_by_sigma,
        sigma_embed_dim=args.sigma_embed_dim,
        norm_by_sigma="norm_by_sigma" in args and args.norm_by_sigma,
        ns=args.ns,
        nv=args.nv,
        distance_embed_dim=args.distance_embed_dim,
        cross_distance_embed_dim=args.cross_distance_embed_dim,
        batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
        use_second_order_repr=args.use_second_order_repr,
        cross_max_distance=args.cross_max_distance,
        dynamic_max_cross=args.dynamic_max_cross,
        smooth_edges=args.smooth_edges if "smooth_edges" in args else False,
        odd_parity=args.odd_parity if "odd_parity" in args else False,
        lm_embedding_type="precomputed",
        confidence_mode=False,
        affinity_prediction=(
            args.affinity_prediction if "affinity_prediction" in args else False
        ),
        parallel=args.parallel if "parallel" in args else 1,
        num_confidence_outputs=(
            len(args.rmsd_classification_cutoff) + 1
            if "rmsd_classification_cutoff" in args
            and isinstance(args.rmsd_classification_cutoff, list)
            else 1
        ),
        atom_num_confidence_outputs=(
            len(args.atom_rmsd_classification_cutoff) + 1
            if "atom_rmsd_classification_cutoff" in args
            and isinstance(args.atom_rmsd_classification_cutoff, list)
            else 1
        ),
        parallel_aggregators=(
            args.parallel_aggregators if "parallel_aggregators" in args else ""
        ),
        fixed_center_conv=(
            not args.not_fixed_center_conv if "not_fixed_center_conv" in args else False
        ),
        no_aminoacid_identities=(
            args.no_aminoacid_identities if "no_aminoacid_identities" in args else False
        ),
        include_miscellaneous_atoms=(
            args.include_miscellaneous_atoms
            if hasattr(args, "include_miscellaneous_atoms")
            else False
        ),
        sh_lmax=args.sh_lmax if "sh_lmax" in args else 2,
        differentiate_convolutions=(
            not args.no_differentiate_convolutions
            if "no_differentiate_convolutions" in args
            else True
        ),
        tp_weights_layers=args.tp_weights_layers if "tp_weights_layers" in args else 2,
        num_prot_emb_layers=(
            args.num_prot_emb_layers if "num_prot_emb_layers" in args else 0
        ),
        reduce_pseudoscalars=(
            args.reduce_pseudoscalars if "reduce_pseudoscalars" in args else False
        ),
        embed_also_ligand=(
            args.embed_also_ligand if "embed_also_ligand" in args else False
        ),
        atom_confidence=(
            args.atom_confidence_loss_weight > 0.0
            if "atom_confidence_loss_weight" in args
            else False
        ),
        sidechain_pred=(
            hasattr(args, "sidechain_loss_weight") and args.sidechain_loss_weight > 0
        )
        or (hasattr(args, "backbone_loss_weight") and args.backbone_loss_weight > 0),
        depthwise_convolution=(
            args.depthwise_convolution
            if hasattr(args, "depthwise_convolution")
            else False
        ),
    )

    if (
        device.type == "cuda"
        and not no_parallel
        and ("dataset" not in args or not args.dataset == "torsional")
    ):
        model = DataParallel(model)

    model.to(device)
    return model


def crop_beyond(complex_graph, cutoff, all_atoms):
    ligand_pos = complex_graph["ligand"].pos
    receptor_pos = complex_graph["receptor"].pos
    residues_to_keep = torch.any(
        torch.sum((ligand_pos.unsqueeze(0) - receptor_pos.unsqueeze(1)) ** 2, -1)
        < cutoff**2,
        dim=1,
    )

    if all_atoms:
        atom_to_res_mapping = complex_graph[
            "atom", "atom_rec_contact", "receptor"
        ].edge_index[1]
        atoms_to_keep = residues_to_keep[atom_to_res_mapping]
        rec_remapper = torch.cumsum(residues_to_keep.long(), dim=0) - 1
        atom_to_res_new_mapping = rec_remapper[atom_to_res_mapping][atoms_to_keep]
        atom_res_edge_index = torch.stack(
            [
                torch.arange(
                    len(atom_to_res_new_mapping), device=atom_to_res_new_mapping.device
                ),
                atom_to_res_new_mapping,
            ]
        )

    complex_graph["receptor"].pos = complex_graph["receptor"].pos[residues_to_keep]
    complex_graph["receptor"].x = complex_graph["receptor"].x[residues_to_keep]
    complex_graph["receptor"].side_chain_vecs = complex_graph[
        "receptor"
    ].side_chain_vecs[residues_to_keep]
    complex_graph["receptor", "rec_contact", "receptor"].edge_index = subgraph(
        residues_to_keep,
        complex_graph["receptor", "rec_contact", "receptor"].edge_index,
        relabel_nodes=True,
    )[0]

    if all_atoms:
        complex_graph["atom"].x = complex_graph["atom"].x[atoms_to_keep]
        complex_graph["atom"].pos = complex_graph["atom"].pos[atoms_to_keep]
        complex_graph["atom", "atom_contact", "atom"].edge_index = subgraph(
            atoms_to_keep,
            complex_graph["atom", "atom_contact", "atom"].edge_index,
            relabel_nodes=True,
        )[0]
        complex_graph["atom", "atom_rec_contact", "receptor"].edge_index = (
            atom_res_edge_index
        )
