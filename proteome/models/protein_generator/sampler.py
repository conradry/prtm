from dataclasses import asdict

import numpy as np
import torch
from tqdm import tqdm

import proteome.models.protein_generator.diff_utils as diff_utils
from proteome import protein
from proteome.common_modules.rosetta.contigs import *
from proteome.common_modules.rosetta.kinematics import get_init_xyz, xyz_to_t2d
from proteome.common_modules.rosetta.util import *
from proteome.constants.residue_constants import restypes_with_x_dash
from proteome.models.protein_generator import config
from proteome.models.protein_generator.calc_dssp import annotate_sse
from proteome.models.protein_generator.diffusion import GaussianDiffusion_SEQDIFF
from proteome.models.protein_generator.potentials import POTENTIALS


class SeqDiffSampler:

    """
    MODULAR SAMPLER FOR SEQUENCE DIFFUSION

    - the goal for modularizing this code is to make it as
      easy as possible to edit and mix functions around

    - in the base implementation here this can handle the standard
      inference mode with default passes through the model, different
      forms of partial diffusion, and linear symmetry

    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: config.InferenceConfig,
        pad_t1d_to_29: bool = False,
    ):
        """
        set args and DEVICE as well as other default params
        """
        self.model = model
        self.cfg = cfg
        self.dssp_dict = {"X": 3, "H": 0, "E": 1, "L": 2}
        self.device = list(self.model.parameters())[0].device

        self.potentials_conf = cfg.potentials_params
        self.contig_conf = cfg.contigmap_params
        self.structure_bias_conf = cfg.structure_bias_params
        self.diffuser_conf = cfg.diffuser_params
        self.hotspot_conf = cfg.hotspot_params
        self.symmetry_conf = cfg.symmetry_params
        self.secondary_structure_conf = cfg.secondary_structure_params

        self.pad_t1d_to_29 = pad_t1d_to_29

    def diffuser_init(self):
        """
        set up diffuser object of GaussianDiffusion_SEQDIFF
        """
        self.diffuser = GaussianDiffusion_SEQDIFF(
            T=self.diffuser_conf.T,
            schedule=self.diffuser_conf.schedule,
            sample_distribution=self.diffuser_conf.sample_distribution,
            sample_distribution_gmm_means=self.diffuser_conf.sample_distribution_gmm_means,
            sample_distribution_gmm_variances=self.diffuser_conf.sample_distribution_gmm_variances,
        )
        self.betas = self.diffuser.betas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

    def make_hotspot_features(self):
        """
        set up hotspot features
        """
        # initialize hotspot features to all 0s
        self.features["hotspot_feat"] = torch.zeros(self.features["L"])

        # if hotspots exist in args then make hotspot features
        if self.hotspot_conf.hotspot_res != None:
            self.features["hotspots"] = [
                (x[0], int(x[1:])) for x in self.hotspot_conf.hotspot_res
            ]
            for n, x in enumerate(self.features["mappings"]["complex_con_ref_pdb_idx"]):
                if x in self.features["hotspots"]:
                    self.features["hotspot_feat"][
                        self.features["mappings"]["complex_con_hal_idx0"][n]
                    ] = 1.0

    def make_dssp_features(self):
        """
        set up dssp features
        """

        # initialize with all zeros
        self.features["dssp_feat"] = torch.zeros(self.features["L"], 4)

        if self.secondary_structure_conf.secondary_structure != None:
            self.features["secondary_structure"] = [
                self.dssp_dict[x.upper()]
                for x in self.secondary_structure_conf.secondary_structure
            ]

            assert (
                len(self.features["secondary_structure"] * self.features["sym"])
                + self.features["cap"] * 2
                == self.features["L"]
            ), f"You have specified a secondary structure string that does not match your design length"

            self.features["dssp_feat"] = torch.nn.functional.one_hot(
                torch.tensor(
                    self.features["cap_dssp"]
                    + self.features["secondary_structure"] * self.features["sym"]
                    + self.features["cap_dssp"]
                ),
                num_classes=4,
            )

        elif self.secondary_structure_conf.dssp_structure != None:
            dssp_xyz = torch.from_numpy(
                self.secondary_structure_conf.dssp_structure.atom_positions
            )
            dssp_sse = annotate_sse(
                np.array(dssp_xyz[:, 1, :].squeeze()), percentage_mask=0
            )
            # we assume binder is chain A
            self.features["dssp_feat"][: dssp_sse.shape[0]] = dssp_sse

        elif sum(asdict(self.structure_bias_conf).values()) > 0.0:
            tmp_mask = (
                torch.rand(self.features["L"]) < self.structure_bias_conf.helix_bias
            )
            self.features["dssp_feat"][tmp_mask, 0] = 1.0

            tmp_mask = (
                torch.rand(self.features["L"]) < self.structure_bias_conf.strand_bias
            )
            self.features["dssp_feat"][tmp_mask, 1] = 1.0

            tmp_mask = (
                torch.rand(self.features["L"]) < self.structure_bias_conf.loop_bias
            )
            self.features["dssp_feat"][tmp_mask, 2] = 1.0

        # contigs get mask label
        self.features["dssp_feat"][self.features["mask_str"][0], 3] = 1.0
        # anything not labeled gets mask label
        mask_index = torch.where(torch.sum(self.features["dssp_feat"], dim=1) == 0)[0]
        self.features["dssp_feat"][mask_index, 3] = 1.0

    def feature_init(self):
        """
        featurize pdb and contigs and choose type of diffusion
        """
        # initialize features dictionary for all example features
        self.features = {}

        # set up params
        self.loader_params = {"MAXCYCLE": self.cfg.n_cycle}

        # symmetry
        self.features["sym"] = self.symmetry_conf.symmetry
        self.features["cap"] = self.symmetry_conf.symmetry_cap
        self.features["cap_dssp"] = [
            self.dssp_dict[x.upper()] for x in "H" * self.features["cap"]
        ]

        assert (self.contig_conf.contigs is None) ^ (
            self.cfg.sequence is None
        ), f"You are specifying contigs ({self.contig_conf.contigs}) and sequence ({self.cfg.sequence})  (or neither), please specify one or the other"

        # initialize trb dictionary
        self.features["trb_d"] = {}

        if self.cfg.reference_structure == None and self.cfg.sequence is not None:
            allowable_aas = [x for x in restypes_with_x_dash[:-1]]
            for x in self.cfg.sequence:
                assert (
                    x in allowable_aas
                ), f"Amino Acid {x} is undefinded, please only use standart 20 AAs"
            self.features["seq"] = torch.tensor(
                [restypes_with_x_dash.index(x) for x in self.cfg.sequence]
            )
            self.features["xyz_t"] = torch.full(
                (1, 1, len(self.cfg.sequence), 27, 3), np.nan
            )
            self.features["mask_str"] = (
                torch.zeros(len(self.cfg.sequence)).long()[None, :].bool()
            )

            # added check for if in partial diffusion mode will mask
            if self.cfg.sampling_temp == 1.0:
                self.features["mask_seq"] = (
                    torch.tensor([0 if x == "X" else 1 for x in self.cfg.sequence])
                    .long()[None, :]
                    .bool()
                )
            else:
                self.features["mask_seq"] = (
                    torch.zeros(len(self.cfg.sequence)).long()[None, :].bool()
                )

            self.features["blank_mask"] = torch.ones(
                self.features["mask_str"].size()[-1]
            )[None, :].bool()
            self.features["idx_pdb"] = torch.tensor(
                [i for i in range(len(self.cfg.sequence))]
            )[None, :]
            conf_1d = torch.ones_like(self.features["seq"])
            conf_1d[~self.features["mask_str"][0]] = 0
            (
                self.features["seq_hot"],
                self.features["msa"],
                self.features["msa_hot"],
                self.features["msa_extra_hot"],
                _,
            ) = MSAFeaturize_fixbb(self.features["seq"][None, :], self.loader_params)

            self.features["t1d"] = TemplFeaturizeFixbb(
                self.features["seq"], conf_1d=conf_1d
            )[None, None, :]
            self.features["seq_hot"] = self.features["seq_hot"].unsqueeze(dim=0)
            self.features["msa"] = self.features["msa"].unsqueeze(dim=0)
            self.features["msa_hot"] = self.features["msa_hot"].unsqueeze(dim=0)
            self.features["msa_extra_hot"] = self.features["msa_extra_hot"].unsqueeze(
                dim=0
            )

            self.max_t = int(self.diffuser_conf.T * self.cfg.sampling_temp)

            self.features["pdb_idx"] = [
                ("A", i + 1) for i in range(len(self.cfg.sequence))
            ]
            self.features["trb_d"]["inpaint_str"] = self.features["mask_str"][0]
            self.features["trb_d"]["inpaint_seq"] = self.features["mask_seq"][0]

        else:
            assert not (
                self.cfg.reference_structure == None and self.cfg.sampling_temp != 1.0
            ), f"You must specify a pdb if attempting to use contigs with partial diffusion, else partially diffuse sequence input"

            if self.cfg.reference_structure == None:
                reference_structure = protein.Protein27(
                    atom_positions=np.zeros((1, 27, 3)),
                    aatype=np.zeros((1,), dtype=np.int32),
                    atom_mask=np.zeros((1, 27)),
                    residue_index=np.arange(1, dtype=np.int32),
                    b_factors=np.zeros((1, 27)),
                    chain_index=np.array([0], dtype=np.int32),
                )
            else:
                reference_structure = self.cfg.reference_structure

            # generate contig map
            self.features["rm"] = ContigMap(
                reference_structure, **asdict(self.contig_conf)
            )

            self.features["mappings"] = self.features["rm"].get_mappings()
            self.features["pdb_idx"] = self.features["rm"].hal

            ### PREPARE FEATURES DEPENDING ON TYPE OF ARGUMENTS SPECIFIED ###

            # FULL DIFFUSION MODE
            if self.cfg.sampling_temp == 1.0:
                # process contigs and generate masks
                self.features["mask_str"] = torch.from_numpy(
                    self.features["rm"].inpaint_str
                )[None, :]
                self.features["mask_seq"] = torch.from_numpy(
                    self.features["rm"].inpaint_seq
                )[None, :]
                self.features["blank_mask"] = torch.ones(
                    self.features["mask_str"].size()[-1]
                )[None, :].bool()

                seq_input = torch.from_numpy(reference_structure.aatype).long()
                xyz_input = torch.from_numpy(reference_structure.atom_positions).float()

                self.features["xyz_t"] = torch.full(
                    (1, 1, len(self.features["rm"].ref), 27, 3), np.nan
                )
                self.features["xyz_t"][
                    :, :, self.features["rm"].hal_idx0, :14, :
                ] = xyz_input[self.features["rm"].ref_idx0, :14, :][None, None, ...]
                self.features["seq"] = torch.full(
                    (1, len(self.features["rm"].ref)), 20
                ).squeeze()
                self.features["seq"][self.features["rm"].hal_idx0] = seq_input[
                    self.features["rm"].ref_idx0
                ]

                # template confidence
                conf_1d = torch.ones_like(self.features["seq"]) * self.cfg.tmpl_conf
                conf_1d[
                    ~self.features["mask_str"][0]
                ] = 0  # zero confidence for places where structure is masked
                seq_masktok = torch.where(
                    self.features["seq"] == 20, 21, self.features["seq"]
                )

                # Get sequence and MSA input features
                (
                    self.features["seq_hot"],
                    self.features["msa"],
                    self.features["msa_hot"],
                    self.features["msa_extra_hot"],
                    _,
                ) = MSAFeaturize_fixbb(seq_masktok[None, :], self.loader_params)
                self.features["t1d"] = TemplFeaturizeFixbb(
                    self.features["seq"], conf_1d=conf_1d
                )[None, None, :]
                self.features["idx_pdb"] = torch.from_numpy(
                    np.array(self.features["rm"].rf)
                ).int()[None, :]
                self.features["seq_hot"] = self.features["seq_hot"].unsqueeze(dim=0)
                self.features["msa"] = self.features["msa"].unsqueeze(dim=0)
                self.features["msa_hot"] = self.features["msa_hot"].unsqueeze(dim=0)
                self.features["msa_extra_hot"] = self.features[
                    "msa_extra_hot"
                ].unsqueeze(dim=0)

                self.max_t = int(self.diffuser_conf.T * self.cfg.sampling_temp)
            else:
                self.features["seq"] = torch.from_numpy(reference_structure.aatype)
                self.features["xyz_t"] = torch.from_numpy(
                    reference_structure.atom_positions
                )[None, None, ...]

                if self.contig_conf.contigs is None:
                    self.features["mask_str"] = (
                        torch.zeros(self.features["xyz_t"].shape[2])
                        .long()[None, :]
                        .bool()
                    )
                    self.features["mask_seq"] = (
                        torch.zeros(self.features["seq"].shape[0])
                        .long()[None, :]
                        .bool()
                    )
                    self.features["blank_mask"] = torch.ones(
                        self.features["mask_str"].size()[-1]
                    )[None, :].bool()
                else:
                    self.features["mask_str"] = torch.from_numpy(
                        self.features["rm"].inpaint_str
                    )[None, :]
                    self.features["mask_seq"] = torch.from_numpy(
                        self.features["rm"].inpaint_seq
                    )[None, :]
                    self.features["blank_mask"] = torch.ones(
                        self.features["mask_str"].size()[-1]
                    )[None, :].bool()

                idx_pdb = []
                chains_used = [reference_structure.chain_index[0]]
                idx_jump = 0
                for i, chain_ix in enumerate(reference_structure.chain_index):
                    if chain_ix not in chains_used:
                        chains_used.append(x[0])
                        idx_jump += 200
                    idx_pdb.append(idx_jump + i)

                self.features["idx_pdb"] = torch.tensor(idx_pdb)[None, :]
                conf_1d = torch.ones_like(self.features["seq"])
                conf_1d[~self.features["mask_str"][0]] = 0
                (
                    self.features["seq_hot"],
                    self.features["msa"],
                    self.features["msa_hot"],
                    self.features["msa_extra_hot"],
                    _,
                ) = MSAFeaturize_fixbb(
                    self.features["seq"][None, :], self.loader_params
                )
                self.features["t1d"] = TemplFeaturizeFixbb(
                    self.features["seq"], conf_1d=conf_1d
                )[None, None, :]
                self.features["seq_hot"] = self.features["seq_hot"].unsqueeze(dim=0)
                self.features["msa"] = self.features["msa"].unsqueeze(dim=0)
                self.features["msa_hot"] = self.features["msa_hot"].unsqueeze(dim=0)
                self.features["msa_extra_hot"] = self.features[
                    "msa_extra_hot"
                ].unsqueeze(dim=0)

                self.max_t = int(self.diffuser_conf.T * self.cfg.sampling_temp)

        # set L
        self.features["L"] = self.features["seq"].shape[0]

    def potential_init(self):
        """
        initialize potential functions being used and return list of potentails
        """

        potentials = self.potentials_conf.potentials
        potential_scales = self.potentials_conf.potential_scales

        self.potential_list = []
        for potential, scale in zip(potentials, potential_scales):
            potential_type = potential._potential_type  # type: ignore
            assert (
                potential_type in POTENTIALS.keys()
            ), f"The potential specified: {potential_type} , does not match into POTENTIALS dictionary in potentials.py"

            self.potential_list.append(
                POTENTIALS[potential_type](
                    potential, self.features["L"], scale, self.device
                )  # type: ignore
            )

        self.use_potentials = True

    def setup(self):
        """
        run init model and init features to get everything prepped to go into model
        """

        # initialize features
        self.feature_init()

        # initialize potential
        if self.potentials_conf.potentials is not None:
            self.potential_init()
        else:
            self.use_potentials = False

        # make hostspot features
        self.make_hotspot_features()

        # make dssp features
        self.make_dssp_features()

        # diffuse sequence and mask features
        (
            self.features["seq"],
            self.features["msa_masked"],
            self.features["msa_full"],
            self.features["xyz_t"],
            self.features["t1d"],
            self.features["seq_diffused"],
        ) = diff_utils.mask_inputs(
            self.features["seq_hot"],
            self.features["msa_hot"],
            self.features["msa_extra_hot"],
            self.features["xyz_t"],
            self.features["t1d"],
            input_seq_mask=self.features["mask_seq"],
            input_str_mask=self.features["mask_str"],
            diffuser=self.diffuser,
            t=self.max_t,
            pad_t1d_to_29=self.pad_t1d_to_29,
            hotspots=self.features["hotspot_feat"],
            dssp=self.features["dssp_feat"],
        )

        # move features to device
        self.features["idx_pdb"] = (
            self.features["idx_pdb"].long().to(self.device, non_blocking=True)
        )  # (B, L)
        self.features["mask_str"] = self.features["mask_str"][None].to(
            self.device, non_blocking=True
        )  # (B, L)
        self.features["xyz_t"] = self.features["xyz_t"][None].to(
            self.device, non_blocking=True
        )
        self.features["t1d"] = self.features["t1d"][None].to(
            self.device, non_blocking=True
        )
        self.features["seq"] = (
            self.features["seq"][None]
            .type(torch.float32)
            .to(self.device, non_blocking=True)
        )
        self.features["msa"] = (
            self.features["msa"].type(torch.float32).to(self.device, non_blocking=True)
        )
        self.features["msa_masked"] = (
            self.features["msa_masked"][None]
            .type(torch.float32)
            .to(self.device, non_blocking=True)
        )
        self.features["msa_full"] = (
            self.features["msa_full"][None]
            .type(torch.float32)
            .to(self.device, non_blocking=True)
        )
        self.ti_dev = torsion_indices.to(self.device, non_blocking=True)
        self.ti_flip = torsion_can_flip.to(self.device, non_blocking=True)
        self.ang_ref = reference_angles.to(self.device, non_blocking=True)
        self.features["xyz_prev"] = torch.clone(self.features["xyz_t"][0])
        self.features["seq_diffused"] = self.features["seq_diffused"][None].to(
            self.device, non_blocking=True
        )
        self.features["B"], _, self.features["N"], self.features["L"] = self.features[
            "msa"
        ].shape
        self.features["t2d"] = xyz_to_t2d(self.features["xyz_t"])

        # get alphas
        self.features["alpha"], self.features["alpha_t"] = diff_utils.get_alphas(
            self.features["t1d"],
            self.features["xyz_t"],
            self.features["B"],
            self.features["L"],
            self.ti_dev,
            self.ti_flip,
            self.ang_ref,
        )

        # processing template coordinates
        self.features["xyz_t"] = get_init_xyz(self.features["xyz_t"])
        self.features["xyz_prev"] = get_init_xyz(
            self.features["xyz_prev"][:, None]
        ).reshape(self.features["B"], self.features["L"], 27, 3)

        # initialize extra features to none
        self.features["xyz"] = None
        self.features["pred_lddt"] = None
        self.features["logit_s"] = None
        self.features["logit_aa_s"] = None
        self.features["best_plddt"] = 0
        self.features["best_pred_lddt"] = torch.zeros_like(self.features["mask_str"])[
            0
        ].float()
        self.features["msa_prev"] = None
        self.features["pair_prev"] = None
        self.features["state_prev"] = None

    def symmetrize_seq(self, x):
        """
        symmetrize x according sym in features
        """
        assert (self.features["L"] - self.features["cap"] * 2) % self.features[
            "sym"
        ] == 0, f"symmetry does not match for input length"
        assert (
            x.shape[0] == self.features["L"]
        ), f"make sure that dimension 0 of input matches to L"

        if self.features["cap"] > 0:
            n_cap = torch.clone(x[: self.features["cap"]])
            c_cap = torch.clone(x[-self.features["cap"] + 1 :])
            sym_x = torch.clone(
                x[self.features["cap"] : self.features["L"] // self.features["sym"]]
            ).repeat(self.features["sym"], 1)

            return torch.cat([n_cap, sym_x, c_cap], dim=0)
        else:
            return torch.clone(x[: self.features["L"] // self.features["sym"]]).repeat(
                self.features["sym"], 1
            )

    def predict_x(self):
        """
        take step using X_t-1 features to predict Xo
        """
        (
            self.features["seq"],
            self.features["xyz"],
            self.features["pred_lddt"],
            self.features["logit_s"],
            self.features["logit_aa_s"],
            self.features["alpha"],
            self.features["msa_prev"],
            self.features["pair_prev"],
            self.features["state_prev"],
        ) = diff_utils.take_step_nostate(
            self.model,
            self.features["msa_masked"],
            self.features["msa_full"],
            self.features["seq"],
            self.features["t1d"],
            self.features["t2d"],
            self.features["idx_pdb"],
            self.cfg.n_cycle,
            self.features["xyz_prev"],
            self.features["alpha"],
            self.features["xyz_t"],
            self.features["alpha_t"],
            self.features["seq_diffused"],
            self.features["msa_prev"],
            self.features["pair_prev"],
            self.features["state_prev"],
        )

    def predict_final_symmetric(self):
        """
        ensure symmetrization with one final prediction
        """
        # take argmaxed seq and make one hot
        if self.cfg.save_best_plddt:
            sym_seq = torch.clone(self.features["best_seq"])
        else:
            sym_seq = torch.clone(self.features["seq"])

        # symmetrize
        sym_seq = self.symmetrize_seq(sym_seq[:, None]).squeeze(-1)

        sym_seq_hot = (
            torch.nn.functional.one_hot(sym_seq, num_classes=22).float() * 2 - 1
        )

        # match other features to seq diffused
        self.features["seq"] = sym_seq[None, None]
        self.features["seq_diffused"] = sym_seq_hot[None]
        self.features["msa_masked"][:, :, :, :, :22] = sym_seq_hot
        self.features["msa_masked"][:, :, :, :, 22:44] = sym_seq_hot
        self.features["msa_full"][:, :, :, :, :22] = sym_seq_hot
        self.features["t1d"][:1, :, :, 22] = 1  # timestep
        self.features["t1d"][
            :1, :, :, 21
        ] = 1  # seq confidence (set to 1 because dont want to change)

        self.predict_x()

        self.features["seq_out"] = torch.permute(self.features["logit_aa_s"][0], (1, 0))
        self.features["best_seq"] = torch.argmax(
            torch.clone(self.features["seq_out"]), dim=-1
        )
        self.features["best_pred_lddt"] = torch.clone(self.features["pred_lddt"])
        self.features["best_xyz"] = torch.clone(self.features["xyz"])
        self.features["best_plddt"] = (
            self.features["pred_lddt"][~self.features["mask_seq"]].mean().item()
        )

    def self_condition_seq(self):
        """
        get previous logits and set at t1d template
        """
        self.features["t1d"][:, :, :, :21] = self.features["logit_aa_s"][
            0, :21, :
        ].permute(1, 0)

    def self_condition_str_scheduled(self):
        """
        unmask random fraction of residues according to timestep
        """
        mask_str = self.features["mask_str"]
        xyz_prev_template = torch.clone(self.features["xyz"])[None]
        self_conditioning_mask = (
            torch.rand(self.features["L"]) < self.diffuser.alphas_cumprod[self.t]
        )
        xyz_prev_template[:, :, ~self_conditioning_mask] = float("nan")
        xyz_prev_template[:, :, self.features["mask_str"][0][0]] = float("nan")
        xyz_prev_template[:, :, :, 3:] = float("nan")
        t2d_sc = xyz_to_t2d(xyz_prev_template)

        xyz_t_sc = torch.zeros_like(self.features["xyz_t"][:, :1])
        xyz_t_sc[:, :, :, :3] = xyz_prev_template[:, :, :, :3]
        xyz_t_sc[:, :, :, 3:] = float("nan")

        t1d_sc = torch.clone(self.features["t1d"][:, :1])
        t1d_sc[:, :, ~self_conditioning_mask] = 0
        t1d_sc[:, :, mask_str[0][0]] = 0

        self.features["t1d"] = torch.cat([self.features["t1d"][:, :1], t1d_sc], dim=1)
        self.features["t2d"] = torch.cat([self.features["t2d"][:, :1], t2d_sc], dim=1)
        self.features["xyz_t"] = torch.cat(
            [self.features["xyz_t"][:, :1], xyz_t_sc], dim=1
        )

        self.features["alpha"], self.features["alpha_t"] = diff_utils.get_alphas(
            self.features["t1d"],
            self.features["xyz_t"],
            self.features["B"],
            self.features["L"],
            self.ti_dev,
            self.ti_flip,
            self.ang_ref,
        )
        self.features["xyz_t"] = get_init_xyz(self.features["xyz_t"])

    def self_condition_str(self):
        """
        conditioining on strucutre in NAR way
        """
        xyz_t_str_sc = torch.zeros_like(self.features["xyz_t"][:, :1])
        xyz_t_str_sc[:, :, :, :3] = torch.clone(self.features["xyz"])[None]
        xyz_t_str_sc[:, :, :, 3:] = float("nan")
        t2d_str_sc = xyz_to_t2d(self.features["xyz_t"])
        t1d_str_sc = torch.clone(self.features["t1d"])

        self.features["xyz_t"] = torch.cat(
            [self.features["xyz_t"], xyz_t_str_sc], dim=1
        )
        self.features["t2d"] = torch.cat([self.features["t2d"], t2d_str_sc], dim=1)
        self.features["t1d"] = torch.cat([self.features["t1d"], t1d_str_sc], dim=1)

    def noise_x(self):
        """
        get X_t-1 from predicted Xo
        """
        # sample x_t-1
        self.features["post_mean"] = self.diffuser.q_sample(
            self.features["seq_out"], self.t, DEVICE=self.device
        )

        if self.features["sym"] > 1:
            self.features["post_mean"] = self.symmetrize_seq(self.features["post_mean"])

        # update seq and masks
        self.features["seq_diffused"][
            0, ~self.features["mask_seq"][0], :21
        ] = self.features["post_mean"][~self.features["mask_seq"][0], ...]
        self.features["seq_diffused"][0, :, 21] = 0.0

        # did not know we were clamping seq
        self.features["seq_diffused"] = torch.clamp(
            self.features["seq_diffused"], min=-3, max=3
        )

        # match other features to seq diffused
        self.features["seq"] = torch.argmax(self.features["seq_diffused"], dim=-1)[None]
        self.features["msa_masked"][:, :, :, :, :22] = self.features["seq_diffused"]
        self.features["msa_masked"][:, :, :, :, 22:44] = self.features["seq_diffused"]
        self.features["msa_full"][:, :, :, :, :22] = self.features["seq_diffused"]
        self.features["t1d"][:1, :, :, 22] = 1 - int(self.t) / self.diffuser_conf.T

    def apply_potentials(self):
        """
        apply potentials
        """

        grads = torch.zeros_like(self.features["seq_out"])
        for p in self.potential_list:
            grads += p.get_gradients(self.features["seq_out"])

        self.features["seq_out"] += grads / len(self.potential_list)

    def generate_sample(self):
        """
        sample from the model

        this function runs the full sampling loop
        """
        # setup example
        self.setup()

        # main sampling loop
        for j in tqdm(range(self.max_t)):
            self.t = torch.tensor(self.max_t - j - 1).to(self.device)

            # run features through the model to get X_o prediction
            self.predict_x()

            # get seq out
            self.features["seq_out"] = torch.permute(
                self.features["logit_aa_s"][0], (1, 0)
            )

            # save best seq
            if self.features["pred_lddt"].mean().item() > self.features["best_plddt"]:
                self.features["best_seq"] = torch.argmax(
                    torch.clone(self.features["seq_out"]), dim=-1
                )
                self.features["best_pred_lddt"] = torch.clone(
                    self.features["pred_lddt"]
                )
                self.features["best_xyz"] = torch.clone(self.features["xyz"])
                self.features["best_plddt"] = self.features["pred_lddt"].mean().item()

            # self condition on sequence
            self.self_condition_seq()

            # self condition on structure
            if self.cfg.scheduled_str_cond:
                self.self_condition_str_scheduled()
            if self.cfg.struc_cond_sc:
                self.self_condition_str()

            # sequence alterations
            if self.cfg.softmax_seqout:
                self.features["seq_out"] = (
                    torch.softmax(self.features["seq_out"], dim=-1) * 2 - 1
                )
            if self.cfg.clamp_seqout:
                self.features["seq_out"] = torch.clamp(
                    self.features["seq_out"],
                    min=-((1 / self.diffuser.alphas_cumprod[self.t]) * 0.25 + 5),
                    max=((1 / self.diffuser.alphas_cumprod[self.t]) * 0.25 + 5),
                )

            # apply potentials
            if self.use_potentials:
                self.apply_potentials()

            # noise to X_t-1
            if self.t != 0:
                self.noise_x()

        # extra pass to ensure symmetrization
        if self.features["sym"] > 1 and self.symmetry_conf.predict_symmetric:
            self.predict_final_symmetric()

        return self.features
