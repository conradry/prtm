from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import tree
from proteome.constants import residue_constants
from proteome.models.design.se3_diffusion import all_atom, config
from proteome.models.design.se3_diffusion import data_utils as du
from proteome.models.design.se3_diffusion import rigid_utils as ru
from proteome.models.design.se3_diffusion.score_network import ScoreNetwork
from proteome.models.design.se3_diffusion.se3_diffuser import SE3Diffuser
from proteome.models.folding.openfold.data import data_transforms

CA_IDX = residue_constants.atom_order["CA"]


def process_chain(design_pdb_feats):
    chain_feats = {
        "aatype": torch.tensor(design_pdb_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(design_pdb_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(design_pdb_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    seq_idx = (
        design_pdb_feats["residue_index"]
        - np.min(design_pdb_feats["residue_index"])
        + 1
    )
    chain_feats["seq_idx"] = seq_idx
    chain_feats["res_mask"] = design_pdb_feats["bb_mask"]
    chain_feats["residue_index"] = design_pdb_feats["residue_index"]
    return chain_feats


def create_pad_feats(pad_amt):
    return {
        "res_mask": torch.ones(pad_amt),
        "fixed_mask": torch.zeros(pad_amt),
        "rigids_impute": torch.zeros((pad_amt, 4, 4)),
        "torsion_impute": torch.zeros((pad_amt, 7, 2)),
    }


class Sampler:
    def __init__(
        self,
        model: ScoreNetwork,
        diffuser: SE3Diffuser,
        diffusion_params: config.DiffusionParams,
    ):
        """Initialize sampler"""
        self.model = model
        self.diffuser = diffuser
        self.diffusion_params = diffusion_params
        self.device = list(model.parameters())[0].device

    def init_data(
        self,
        *,
        rigids_impute,
        torsion_impute,
        fixed_mask,
        res_mask,
    ):
        num_res = res_mask.shape[0]
        diffuse_mask = (1 - fixed_mask) * res_mask
        fixed_mask = fixed_mask * res_mask

        ref_sample = self.diffuser.sample_ref(
            n_samples=num_res,
            rigids_impute=rigids_impute,
            diffuse_mask=diffuse_mask,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, num_res + 1)
        init_feats = {
            "res_mask": res_mask,
            "seq_idx": res_idx * res_mask,
            "fixed_mask": fixed_mask,
            "torsion_angles_sin_cos": torsion_impute,
            "sc_ca_t": torch.zeros_like(rigids_impute.get_trans()),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(lambda x: x[None].to(self.device), init_feats)
        return init_feats

    def _set_t_feats(self, feats, t, t_placeholder):
        feats["t"] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats["rot_score_scaling"] = rot_score_scaling * t_placeholder
        feats["trans_score_scaling"] = trans_score_scaling * t_placeholder
        return feats

    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
        return batch

    @torch.no_grad()
    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=False,
        self_condition=True,
        noise_scale=1.0,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = deepcopy(data_init)
        if sample_feats["rigids_t"].ndim == 2:
            t_placeholder = torch.ones((1,)).to(self.device)
        else:
            t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(
                self.device
            )

        if num_t is None:
            num_t = self.diffusion_params.num_t
        if min_t is None:
            min_t = self.diffusion_param.min_t

        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1 / num_t
        all_rigids = [du.move_to_np(deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []

        if self.model.embed_self_conditioning and self_condition:
            sample_feats = self._set_t_feats(
                sample_feats, reverse_steps[0], t_placeholder
            )
            sample_feats = self._self_conditioning(sample_feats)

        for t in reverse_steps:
            if t > min_t:
                sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                model_out = self.model(sample_feats)
                rot_score = model_out["rot_score"]
                trans_score = model_out["trans_score"]
                rigid_pred = model_out["rigids"]
                if self.model.embed_self_conditioning:
                    sample_feats["sc_ca_t"] = rigid_pred[..., 4:]

                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                diffuse_mask = (1 - sample_feats["fixed_mask"]) * sample_feats[
                    "res_mask"
                ]
                rigids_t = self.diffuser.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_score=du.move_to_np(rot_score),
                    trans_score=du.move_to_np(trans_score),
                    diffuse_mask=du.move_to_np(diffuse_mask),
                    t=t,
                    dt=dt,
                    center=center,
                    noise_scale=noise_scale,
                )
            else:
                model_out = self.model(sample_feats)
                rigids_t = ru.Rigid.from_tensor_7(model_out["rigids"])

            sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(self.device)
            if aux_traj:
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

            # Calculate x0 prediction derived from score predictions.
            gt_trans_0 = sample_feats["rigids_t"][..., 4:]
            pred_trans_0 = rigid_pred[..., 4:]
            trans_pred_0 = (
                diffuse_mask[..., None] * pred_trans_0
                + fixed_mask[..., None] * gt_trans_0
            )
            psi_pred = model_out["psi"]
            if aux_traj:
                atom37_0 = all_atom.compute_backbone(
                    ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                )[0]
                all_bb_0_pred.append(du.move_to_np(atom37_0))
                all_trans_0_pred.append(du.move_to_np(trans_pred_0))
            atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
            all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            "prot_traj": all_bb_prots,
        }
        if aux_traj:
            ret["rigid_traj"] = all_rigids
            ret["trans_traj"] = all_trans_0_pred
            ret["psi_pred"] = psi_pred[None]
            ret["rigid_0_traj"] = all_bb_0_pred

        return ret

    def sample(self, sample_length):
        """Sample based on length.

        Args:
            sample_length: length to sample

        Returns:
            Sample outputs. See train_se3_diffusion.inference_fn.
        """
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        # Initialize data
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, sample_length + 1)
        init_feats = {
            "res_mask": res_mask,
            "seq_idx": res_idx,
            "fixed_mask": fixed_mask,
            "torsion_angles_sin_cos": np.zeros((sample_length, 7, 2)),
            "sc_ca_t": np.zeros((sample_length, 3)),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(lambda x: x[None].to(self.device), init_feats)

        # Run inference
        sample_out = self.inference_fn(
            init_feats,
            num_t=self.diffusion_params.num_t,
            min_t=self.diffusion_params.min_t,
            aux_traj=True,
            noise_scale=self.diffusion_params.noise_scale,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)
