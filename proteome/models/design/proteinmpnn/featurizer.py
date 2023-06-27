from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from proteome import protein
from proteome.constants.residue_constants import proteinmppn_restypes
from proteome.models.design.proteinmpnn.config import TiedFeaturizeOutput


def get_sequence_scores(S, log_probs, mask):
    """Negative log probabilities"""
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def decode_sequence(S, mask):
    return "".join(
        [proteinmppn_restypes[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0]
    )


def featurize_protein(
    structure: protein.DesignableProtein,
    padded_sequence_length: int,
    visible_chains: List[int],
    masked_chains: List[int],
) -> Dict[str, np.ndarray]:
    """Featurizes a protein for input to the MPNN."""
    all_chains = masked_chains + visible_chains
    sequence_length = len(structure.aatype)

    x_chain_list = []
    chain_mask_list = []
    chain_seq_list = []
    chain_encoding_list = []
    letter_list = []
    global_idx_start_list = [0]
    masked_chain_length_list = []
    fixed_position_mask_list = []
    omit_AA_mask_list = []
    pssm_coef_list = []
    pssm_bias_list = []
    pssm_log_odds_list = []
    bias_by_res_list = []
    l0 = 0

    residue_idx = -100 * np.ones([padded_sequence_length], dtype=np.int32)
    for chain_pos, chain_id in enumerate(all_chains, 0):
        chain_seq = structure.aatype[structure.chain_index == chain_id]
        chain_length = len(chain_seq)
        x_chain = structure.atom_positions[structure.chain_index == chain_id]
        assert x_chain.shape[1] in [1, 4], "Only 1 or 4 atoms per residue"

        if chain_id in visible_chains:
            chain_mask = np.zeros(chain_length)  # 0.0 for unmasked
        elif chain_id in masked_chains:
            chain_mask = np.ones(chain_length)  # 1.0 for masked
            masked_chain_length_list.append(chain_length)
        chain_encoding = (chain_pos + 1) * np.ones(np.array(chain_mask).shape[0])

        residue_idx[l0 : l0 + chain_length] = (100 * chain_pos) + np.arange(
            l0, l0 + chain_length
        )
        l0 += chain_length

        letter_list.append(chain_id)
        global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
        x_chain_list.append(x_chain)
        chain_mask_list.append(chain_mask)
        chain_seq_list.append(chain_seq)
        chain_encoding_list.append(chain_encoding)

        fixed_position_mask = structure.design_mask[structure.chain_index == chain_id]
        fixed_position_mask_list.append(fixed_position_mask)
        omit_AA_mask_temp = structure.design_aatype_mask[
            structure.chain_index == chain_id
        ]
        omit_AA_mask_list.append(omit_AA_mask_temp)
        pssm_coef = structure.pssm_coef[structure.chain_index == chain_id]
        pssm_coef_list.append(pssm_coef)
        pssm_bias = structure.pssm_bias[structure.chain_index == chain_id]
        pssm_bias_list.append(pssm_bias)
        pssm_log_odds = structure.pssm_log_odds[structure.chain_index == chain_id]
        pssm_log_odds_list.append(pssm_log_odds)
        bias_by_res = structure.bias_per_residue[structure.chain_index == chain_id]
        bias_by_res_list.append(bias_by_res)

    letter_list_np = np.array(letter_list)
    tied_pos_list_of_lists = []
    tied_beta = np.ones(sequence_length)
    if structure.tied_positions is not None:
        tied_pos_list = structure.tied_positions
        for tied_item in tied_pos_list:
            one_list = []
            for k, v in tied_item.items():
                start_idx = global_idx_start_list[
                    np.argwhere(letter_list_np == k)[0][0]
                ]
                if isinstance(v[0], list):
                    for v_count in range(len(v[0])):
                        one_list.append(
                            start_idx + v[0][v_count] - 1
                        )  # make 0 to be the first
                        tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                else:
                    for v_ in v:
                        one_list.append(start_idx + v_ - 1)  # make 0 to be the first
            tied_pos_list_of_lists.append(one_list)

    x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
    all_sequence = np.concatenate(chain_seq_list)
    m = np.concatenate(
        chain_mask_list, 0
    )  # [L,], 1.0 for places that need to be predicted
    chain_encoding = np.concatenate(chain_encoding_list, 0)
    m_pos = np.concatenate(
        fixed_position_mask_list, 0
    )  # [L,], 1.0 for places that need to be predicted
    omit_AA_mask = np.concatenate(omit_AA_mask_list, 0)

    pssm_coef_ = np.concatenate(
        pssm_coef_list, 0
    )  # [L,], 1.0 for places that need to be predicted
    pssm_bias_ = np.concatenate(
        pssm_bias_list, 0
    )  # [L,], 1.0 for places that need to be predicted
    pssm_log_odds_ = np.concatenate(
        pssm_log_odds_list, 0
    )  # [L,], 1.0 for places that need to be predicted

    bias_by_res_ = np.concatenate(
        bias_by_res_list, 0
    )  # [L,21], 0.0 for places where AA frequencies don't need to be tweaked

    features_dict = {}
    features_dict["atom_positions"] = x
    features_dict["residue_idx"] = residue_idx
    features_dict["omit_AA_mask"] = omit_AA_mask
    features_dict["chain_M"] = m
    features_dict["chain_M_pos"] = m_pos
    features_dict["chain_encoding_all"] = chain_encoding
    features_dict["pssm_coef_all"] = pssm_coef_
    features_dict["pssm_bias_all"] = pssm_bias_
    features_dict["pssm_log_odds_all"] = pssm_log_odds_
    features_dict["bias_by_res_all"] = bias_by_res_
    features_dict["tied_beta"] = tied_beta
    features_dict["sequence"] = all_sequence

    padding = padded_sequence_length - len(all_sequence)
    for k, v in features_dict.items():
        constant = np.nan if k == "atom_positions" else 0.0
        padding_tuples = [(0, padding)] + [(0, 0)] * (v.ndim - 1)
        features_dict[k] = np.pad(
            v, padding_tuples, "constant", constant_values=(constant,)
        )

    return features_dict, letter_list, masked_chain_length_list, tied_pos_list_of_lists


def tied_featurize(
    batch: List[protein.DesignableProtein],
    device,
    chain_dict=None,
    ca_only=False,
):
    """Pack and pad batch into torch tensors"""
    assert chain_dict is None, "chain_dict is not supported yet"
    B = len(batch)
    lengths = np.array([len(b.aatype) for b in batch], dtype=np.int32)
    L_max = max(lengths)

    feature_dicts: List[Dict[str, np.ndarray]] = []
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    for structure in batch:
        # FIXME: This will break if there is a chain_dict
        if chain_dict is not None:
            masked_chains, visible_chains = chain_dict[structure.chain_index[0]]
        else:
            masked_chains = list(np.unique(structure.chain_index))
            visible_chains = []

        masked_chains.sort()  # sort masked_chains
        visible_chains.sort()  # sort visible_chains

        (
            features_dict,
            letter_list,
            masked_chain_length_list,
            tied_pos_list_of_lists,
        ) = featurize_protein(structure, L_max, visible_chains, masked_chains)
        feature_dicts.append(features_dict)
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_chains)
        masked_list_list.append(masked_chains)
        masked_chain_length_list_list.append(masked_chain_length_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

    # Stack all features on the batch dimension
    features_dict_batch = {}
    for k in feature_dicts[0].keys():
        features_dict_batch[k] = np.stack([d[k] for d in feature_dicts], axis=0)

    isnan = np.isnan(features_dict_batch["atom_positions"])
    features_dict_batch["mask"] = np.isfinite(
        np.sum(features_dict_batch["atom_positions"], (2, 3))
    ).astype(np.float32)
    features_dict_batch["atom_positions"][isnan] = 0.0

    jumps = (
        (
            features_dict_batch["residue_idx"][:, 1:]
            - features_dict_batch["residue_idx"][:, :-1]
        )
        == 1
    ).astype(np.float32)
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    features_dict_batch["dihedral_mask"] = np.concatenate(
        [phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1
    )

    # Convert all arrays in features to device tensors
    for k, v in features_dict_batch.items():
        if k in ["chain_encoding_all", "residue_idx", "sequence"]:
            dtype = torch.long
        else:
            dtype = torch.float32
        features_dict_batch[k] = torch.from_numpy(v).to(dtype=dtype, device=device)

    if ca_only:
        features_dict_batch["atom_positions"] = features_dict_batch["atom_positions"][
            :, :, 0
        ]

    return TiedFeaturizeOutput(
        **features_dict_batch,
        lengths=lengths,
        letter_list_list=letter_list_list,
        visible_list_list=visible_list_list,
        masked_list_list=masked_list_list,
        masked_chain_length_list_list=masked_chain_length_list_list,
        tied_pos_list_of_lists_list=tied_pos_list_of_lists_list,
    )
