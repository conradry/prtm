import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as nn
from proteome import protein
from proteome.models.design.rfdiffusion import config
from proteome.models.design.rfdiffusion import inference_utils as iu
from proteome.models.design.rfdiffusion import symmetry, util
from proteome.models.design.rfdiffusion.chemical import seq2chars
from proteome.models.design.rfdiffusion.contigs import ContigMap
from proteome.models.design.rfdiffusion.diffusion import Diffuser
from proteome.models.design.rfdiffusion.kinematics import get_init_xyz, xyz_to_t2d
from proteome.models.design.rfdiffusion.potentials_manager import PotentialManager
from proteome.models.design.rfdiffusion.rosettafold_model import RoseTTAFoldModule
from proteome.models.design.rfdiffusion.util_module import ComputeAllAtomCoords

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

TOR_INDICES = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES = util.reference_angles


class Sampler:
    def __init__(
        self,
        model: RoseTTAFoldModule,
        diffuser_conf: config.DiffuserConfig,
        preprocess_conf: config.PreprocessConfig,
        inference_conf: config.InferenceConfig,
        contigmap_conf: config.ContigMap = config.ContigMap(),
        denoiser_conf: config.DenoiserParams = config.DenoiserParams(),
        ppi_conf: config.PPIParams = config.PPIParams(),
        potential_conf: config.PotentialsParams = config.PotentialsParams(),
        symmetry_conf: config.SymmetryParams = config.SymmetryParams(),
    ):
        """
        Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.model = model
        self.inf_conf = inference_conf
        self.contig_conf = contigmap_conf
        self.denoiser_conf = denoiser_conf
        self.ppi_conf = ppi_conf
        self.potential_conf = potential_conf
        self.diffuser_conf = diffuser_conf
        self.preprocess_conf = preprocess_conf
        self.symmetry_conf = symmetry_conf

        self.device = list(self.model.parameters())[0].device
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize sampler.
        Args:
            conf: Configuration

        - Selects appropriate model from input
        - Assembles Config from model checkpoint and command line overrides

        """
        # Initialize helper objects
        schedule_directory = f"{SCRIPT_DIR}/../../schedules"
        os.makedirs(schedule_directory, exist_ok=True)
        self.diffuser = Diffuser(
            **asdict(self.diffuser_conf), cache_dir=schedule_directory
        )

        ###########################
        ### Initialise Symmetry ###
        ###########################

        if self.symmetry_conf.symmetry is not None:
            self.symmetry = symmetry.SymGen(
                self.symmetry_conf.symmetry,
                self.symmetry_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius,
            )
        else:
            self.symmetry = None

        self.allatom = ComputeAllAtomCoords().to(self.device)

        self.target_feats = iu.process_target(
            self.inf_conf.input_structure, center=False
        )
        self.chain_idx = None

        ##############################
        ### Handle Partial Noising ###
        ##############################

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)

    @property
    def T(self):
        """
        Return the maximum number of timesteps
        that this design protocol will perform.

        Output:
            T (int): The maximum number of timesteps to perform
        """
        return self.diffuser_conf.T

    def construct_contig(self, target_feats):
        """
        Construct contig class describing the protein to be generated
        """
        contig_dict = asdict(self.contig_conf) if self.contig_conf else {}
        return ContigMap(target_feats, **contig_dict)

    def construct_denoiser(self, L, visible):
        """Make length-specific denoiser."""
        denoise_kwargs = asdict(self.diffuser_conf)
        denoise_kwargs.update(asdict(self.denoiser_conf))
        denoise_kwargs.update(
            {
                "L": L,
                "diffuser": self.diffuser,
                "potential_manager": self.potential_manager,
                "visible": visible,
            }
        )
        return iu.Denoise(**denoise_kwargs)

    def sample_init(self):
        """
        Initial features to start the sampling process.

        Modify signature and function body for different initialization
        based on the config.

        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """

        #######################
        ### Parse input pdb ###
        #######################

        self.target_feats = iu.process_target(
            self.inf_conf.input_structure, center=False
        )

        ################################
        ### Generate specific contig ###
        ################################

        # Generate a specific contig from the range of possibilities specified at input

        self.contig_map = self.construct_contig(self.target_feats)
        self.mappings = self.contig_map.get_mappings()
        self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None, :]
        self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None, :]
        self.binderlen = len(self.contig_map.inpaint)

        ####################
        ### Get Hotspots ###
        ####################

        self.hotspot_0idx = iu.get_idx0_hotspots(
            self.mappings, self.ppi_conf, self.binderlen
        )

        #####################################
        ### Initialise Potentials Manager ###
        #####################################

        self.potential_manager = PotentialManager(
            self.potential_conf,
            self.ppi_conf,
            self.diffuser_conf,
            self.inf_conf,
            self.hotspot_0idx,
            self.binderlen,
        )

        ###################################
        ### Initialize other attributes ###
        ###################################

        xyz_27 = self.target_feats["xyz_27"]
        mask_27 = self.target_feats["mask_27"]
        seq_orig = self.target_feats["seq"]
        L_mapped = len(self.contig_map.ref)
        contig_map = self.contig_map

        self.diffusion_mask = self.mask_str
        self.chain_idx = ["A" if i < self.binderlen else "B" for i in range(L_mapped)]

        ####################################
        ### Generate initial coordinates ###
        ####################################

        if self.diffuser_conf.partial_T:
            assert (
                xyz_27.shape[0] == L_mapped
            ), f"there must be a coordinate in the input PDB for \
                    each residue implied by the contig string for partial diffusion.  length of \
                    input PDB != length of contig string: {xyz_27.shape[0]} != {L_mapped}"
            assert (
                contig_map.hal_idx0 == contig_map.ref_idx0
            ), f"for partial diffusion there can \
                    be no offset between the index of a residue in the input and the index of the \
                    residue in the output, {contig_map.hal_idx0} != {contig_map.ref_idx0}"
            # Partially diffusing from a known structure
            xyz_mapped = xyz_27
            atom_mask_mapped = mask_27
        else:
            # Fully diffusing from points initialised at the origin
            # adjust size of input xt according to residue map
            xyz_mapped = torch.full((1, 1, L_mapped, 27, 3), np.nan)
            xyz_mapped[:, :, contig_map.hal_idx0, ...] = xyz_27[
                contig_map.ref_idx0, ...
            ]
            xyz_motif_prealign = xyz_mapped.clone()
            self.motif_com = xyz_27[contig_map.ref_idx0, 1].mean(dim=0)
            xyz_mapped = get_init_xyz(xyz_mapped).squeeze()
            # adjust the size of the input atom map
            atom_mask_mapped = torch.full((L_mapped, 27), False)
            atom_mask_mapped[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

        # Diffuse the contig-mapped coordinates
        if self.diffuser_conf.partial_T:
            assert (
                self.diffuser_conf.partial_T <= self.diffuser_conf.T
            ), "Partial_T must be less than T"
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input + 1)

        #################################
        ### Generate initial sequence ###
        #################################

        seq_t = torch.full((1, L_mapped), 21).squeeze()  # 21 is the mask token
        seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]

        # Unmask sequence if desired
        if self.contig_conf.provide_seq is not None:
            seq_t[self.mask_seq.squeeze()] = seq_orig[self.mask_seq.squeeze()]

        seq_t[~self.mask_seq.squeeze()] = 21
        seq_t = torch.nn.functional.one_hot(seq_t, num_classes=22).float()  # [L,22]
        seq_orig = torch.nn.functional.one_hot(
            seq_orig, num_classes=22
        ).float()  # [L,22]

        fa_stack, _ = self.diffuser.diffuse_pose(
            xyz_mapped,
            torch.clone(seq_t),
            atom_mask_mapped.squeeze(),
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
        )
        xT = fa_stack[-1].squeeze()[:, :14, :]
        xt = torch.clone(xT)

        self.denoiser = self.construct_denoiser(
            len(self.contig_map.ref), visible=self.mask_seq.squeeze()
        )

        ######################
        ### Apply Symmetry ###
        ######################

        if self.symmetry is not None:
            xt, seq_t = self.symmetry.apply_symmetry(xt, seq_t)

        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None

        #########################################
        ### Parse ligand for ligand potential ###
        #########################################

        if self.potential_conf.guiding_potentials is not None:
            if any(
                list(
                    filter(
                        lambda x: "substrate_contacts" in x,
                        self.potential_conf.guiding_potentials,
                    )
                )
            ):
                assert (
                    len(self.target_feats["xyz_het"]) > 0
                ), "If you're using the Substrate Contact potential, \
                        you need to make sure there's a ligand in the input_pdb file!"
                het_names = np.array(
                    [i["name"].strip() for i in self.target_feats["info_het"]]
                )
                xyz_het = self.target_feats["xyz_het"][
                    het_names == self.potential_conf.substrate
                ]
                xyz_het = torch.from_numpy(xyz_het)
                assert (
                    xyz_het.shape[0] > 0
                ), f"expected >0 heteroatoms from ligand with name {self._conf.potentials.substrate}"
                xyz_motif_prealign = xyz_motif_prealign[0, 0][
                    self.diffusion_mask.squeeze()
                ]
                for pot in self.potential_manager.potentials_to_apply:
                    pot.motif_substrate_atoms = xyz_het
                    pot.diffusion_mask = self.diffusion_mask.squeeze()
                    pot.xyz_motif = xyz_motif_prealign
                    pot.diffuser = self.diffuser
        return xt, seq_t

    def _preprocess(self, seq, xyz_t, t):
        """
        Function to prepare inputs to diffusion model

            seq (L,22) one-hot sequence

            msa_masked (1,1,L,48)

            msa_full (1,1,L,25)

            xyz_t (L,14,3) template crds (diffused)

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)

                MODEL SPECIFIC:
                - contacting residues: for ppi. Target residues in contact with binder (1)
                - empty feature (legacy) (1)
                - ss (H, E, L, MASK) (4)

            t2d (1, L, L, 45)
                - last plane is block adjacency
        """

        L = seq.shape[0]
        T = self.T

        ##################
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1, 1, L, 48))
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0

        ################
        ### msa_full ###
        ################
        msa_full = torch.zeros((1, 1, L, 25))
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0

        ###########
        ### t1d ###
        ###########

        # Here we need to go from one hot with 22 classes to one hot with 21 classes (last plane is missing token)
        t1d = torch.zeros((1, 1, L, 21))

        seqt1d = torch.clone(seq)
        for idx in range(L):
            if seqt1d[idx, 21] == 1:
                seqt1d[idx, 20] = 1
                seqt1d[idx, 21] = 0

        t1d[:, :, :, :21] = seqt1d[None, None, :, :21]

        # Set timestep feature to 1 where diffusion mask is True, else 1-t/T
        timefeature = torch.zeros((L)).float()
        timefeature[self.mask_str.squeeze()] = 1
        timefeature[~self.mask_str.squeeze()] = 1 - t / self.T
        timefeature = timefeature[None, None, ..., None]

        t1d = torch.cat((t1d, timefeature), dim=-1).float()

        #############
        ### xyz_t ###
        #############
        if self.preprocess_conf.sidechain_input:
            xyz_t[torch.where(seq == 21, True, False), 3:, :] = float("nan")
        else:
            xyz_t[~self.mask_str.squeeze(), 3:, :] = float("nan")

        xyz_t = xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1, 1, L, 13, 3), float("nan"))), dim=3)

        ###########
        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_t)

        ###########
        ### idx ###
        ###########
        idx = torch.tensor(self.contig_map.rf)[None]

        ###############
        ### alpha_t ###
        ###############
        seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
        alpha, _, alpha_mask, _ = util.get_torsions(
            xyz_t.reshape(-1, L, 27, 3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES
        )
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1, -1, L, 10, 2)
        alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        # put tensors on device
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        seq = seq.to(self.device)
        xyz_t = xyz_t.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)

        ######################
        ### added_features ###
        ######################
        if self.preprocess_conf.d_t1d >= 24:  # add hotspot residues
            hotspot_tens = torch.zeros(L).float()
            if self.ppi_conf.hotspot_res is None:
                print(
                    "WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots.\
                         If you're doing monomer diffusion this is fine"
                )
                hotspot_idx = []
            else:
                hotspots = [(i[0], int(i[1:])) for i in self.ppi_conf.hotspot_res]
                hotspot_idx = []
                for i, res in enumerate(self.contig_map.con_ref_pdb_idx):
                    if res in hotspots:
                        hotspot_idx.append(self.contig_map.hal_idx0[i])
                hotspot_tens[hotspot_idx] = 1.0

            # Add blank (legacy) feature and hotspot tensor
            t1d = torch.cat(
                (
                    t1d,
                    torch.zeros_like(t1d[..., :1]),
                    hotspot_tens[None, None, ..., None].to(self.device),
                ),
                dim=-1,
            )

        return (
            msa_masked,
            msa_full,
            seq[None],
            torch.squeeze(xyz_t, dim=0),
            idx,
            t1d,
            t2d,
            xyz_t,
            alpha_t,
        )

    def sample_step(self, *, t, x_t, seq_init, final_step):
        raise NotImplementedError


class SelfConditioning(Sampler):
    """
    Model Runner for self conditioning
    pX0[t+1] is provided as a template input to the model at time t
    """

    def sample_step(self, *, t, x_t, seq_init, final_step):
        """
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The sequence to the next step (== seq_init)
            plddt: (L, 1) Predicted lDDT of x0.
        """

        (
            msa_masked,
            msa_full,
            seq_in,
            xt_in,
            idx_pdb,
            t1d,
            t2d,
            xyz_t,
            alpha_t,
        ) = self._preprocess(seq_init, x_t, t)
        B, N, L = xyz_t.shape[:3]

        ##################################
        ######## Str Self Cond ###########
        ##################################
        if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T):
            zeros = torch.zeros(B, 1, L, 24, 3).float().to(xyz_t.device)
            xyz_t = torch.cat(
                (self.prev_pred.unsqueeze(1), zeros), dim=-2
            )  # [B,T,L,27,3]
            t2d_44 = xyz_to_t2d(xyz_t)  # [B,T,L,L,44]
        else:
            xyz_t = torch.zeros_like(xyz_t)
            t2d_44 = torch.zeros_like(t2d[..., :44])
        # No effect if t2d is only dim 44
        t2d[..., :44] = t2d_44

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        ####################
        ### Forward Pass ###
        ####################

        with torch.no_grad():
            msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
                msa_masked,
                msa_full,
                seq_in,
                xt_in,
                idx_pdb,
                t1d=t1d,
                t2d=t2d,
                xyz_t=xyz_t,
                alpha_t=alpha_t,
                msa_prev=None,
                pair_prev=None,
                state_prev=None,
                t=torch.tensor(t),
                return_infer=True,
                motif_mask=self.diffusion_mask.squeeze().to(self.device),
            )

            if self.symmetry is not None and self.inf_conf.symmetric_self_cond:
                px0 = self.symmetrise_prev_pred(px0=px0, seq_in=seq_in, alpha=alpha)[
                    :, :, :3
                ]

        self.prev_pred = torch.clone(px0)

        # prediction of X0
        _, px0 = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0 = px0.squeeze()[:, :14]

        ###########################
        ### Generate Next Input ###
        ###########################

        seq_t_1 = torch.clone(seq_init)
        if t > final_step:
            x_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            )
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            px0 = px0.to(x_t.device)

        ######################
        ### Apply symmetry ###
        ######################
        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)

        return px0, x_t_1, seq_t_1, plddt

    def symmetrise_prev_pred(self, px0, seq_in, alpha):
        """
        Method for symmetrising px0 output for self-conditioning
        """
        _, px0_aa = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0_sym, _ = self.symmetry.apply_symmetry(
            px0_aa.to("cpu").squeeze()[:, :14],
            torch.argmax(seq_in, dim=-1).squeeze().to("cpu"),
        )
        px0_sym = px0_sym[None].to(self.device)
        return px0_sym


class ScaffoldedSampler(SelfConditioning):
    """
    Model Runner for Scaffold-Constrained diffusion
    """

    def __init__(
        self,
        model: RoseTTAFoldModule,
        diffuser_conf: config.DiffuserConfig,
        preprocess_conf: config.PreprocessConfig,
        inference_conf: config.InferenceConfig,
        contigmap_conf: config.ContigMap = config.ContigMap(),
        denoiser_conf: config.DenoiserParams = config.DenoiserParams(),
        ppi_conf: config.PPIParams = config.PPIParams(),
        potential_conf: config.PotentialsParams = config.PotentialsParams(),
        symmetry_conf: config.SymmetryParams = config.SymmetryParams(),
        scaffold_conf: config.ScaffoldGuidedParams = config.ScaffoldGuidedParams(),
    ):
        super().__init__(
            model=model,
            diffuser_conf=diffuser_conf,
            preprocess_conf=preprocess_conf,
            inference_conf=inference_conf,
            contigmap_conf=contigmap_conf,
            denoiser_conf=denoiser_conf,
            ppi_conf=ppi_conf,
            potential_conf=potential_conf,
            symmetry_conf=symmetry_conf,
        )
        self.scaffold_conf = scaffold_conf
        self.blockadjacency = iu.BlockAdjacency(
            scaffold_conf, inference_conf.num_designs
        )

        #################################################
        ### Initialize target, if doing binder design ###
        #################################################

        if scaffold_conf.target_pdb_str:
            struct = protein.from_pdb_string(
                scaffold_conf.target_pdb_str, atom14_format=True, parse_hetatom=True
            )
            self.target = iu.Target(
                struct, scaffold_conf.contig_crop, ppi_conf.hotspot_res
            )
            self.target_struct = self.target.get_target()

            if (
                scaffold_conf.target_ss is not False
                or scaffold_conf.target_adj is not False
            ):
                target_ss, target_adj = self.blockadjacency.get_ss_adj(
                    scaffold_conf.target_pdb_str
                )

            if scaffold_conf.target_ss is not False:
                self.target_ss = torch.nn.functional.one_hot(
                    target_ss.long(), num_classes=4
                )
                if self.scaffold_conf.contig_crop is not None:
                    self.target_ss = self.target_ss[self.target.crop_mask]

            if scaffold_conf.target_adj is not None:
                self.target_adj = torch.nn.functional.one_hot(
                    target_adj.long(), num_classes=3
                )
                if self.scaffold_conf.contig_crop is not None:
                    self.target_adj = self.target_adj[self.target.crop_mask]
                    self.target_adj = self.target_adj[:, self.target.crop_mask]

        else:
            self.target = None
            self.target_struct = None

    def sample_init(self):
        """
        Wrapper method for taking secondary structure + adj, and outputting xt, seq_t
        """

        ##########################
        ### Process Fold Input ###
        ##########################
        self.L, self.ss, self.adj = self.blockadjacency.get_scaffold()
        self.adj = nn.one_hot(self.adj.long(), num_classes=3)

        ##############################
        ### Auto-contig generation ###
        ##############################

        if self.contig_conf.contigs is None:
            # process target
            xT = torch.full((self.L, 27, 3), np.nan)
            xT = get_init_xyz(xT[None, None]).squeeze()
            seq_T = torch.full((self.L,), 21)
            self.diffusion_mask = torch.full((self.L,), False)
            atom_mask = torch.full((self.L, 27), False)
            self.binderlen = self.L

            if self.target:
                target_L = len(self.target_struct.atom_positions)
                # xyz
                target_xyz = torch.full((target_L, 27, 3), np.nan)
                target_xyz[:, :14, :] = torch.from_numpy(
                    self.target_struct.atom_positions
                )
                xT = torch.cat((xT, target_xyz), dim=0)
                # seq
                seq_T = torch.cat(
                    (seq_T, torch.from_numpy(self.target_struct.aatype)), dim=0
                )
                # diffusion mask
                self.diffusion_mask = torch.cat(
                    (self.diffusion_mask, torch.full((target_L,), True)), dim=0
                )
                # atom mask
                mask_27 = torch.full((target_L, 27), False)
                mask_27[:, :14] = torch.from_numpy(self.target_struct.atom_mask)
                atom_mask = torch.cat((atom_mask, mask_27), dim=0)
                self.L += target_L
                # generate contigmap object
                pdb_idx = list(
                    zip(
                        [
                            protein.PDB_CHAIN_IDS[i]
                            for i in self.target_struct.chain_index
                        ],
                        self.target_struct.residue_index,
                    )
                )
                contig = []
                for idx, i in enumerate(pdb_idx[:-1]):
                    if idx == 0:
                        start = i[1]
                    if i[1] + 1 != pdb_idx[idx + 1][1] or i[0] != pdb_idx[idx + 1][0]:
                        contig.append(f"{i[0]}{start}-{i[1]}/0 ")
                        start = pdb_idx[idx + 1][1]

                contig.append(f"{pdb_idx[-1][0]}{start}-{pdb_idx[-1][1]}/0 ")
                contig.append(f"{self.binderlen}-{self.binderlen}")
                contig = ["".join(contig)]
            else:
                contig = [f"{self.binderlen}-{self.binderlen}"]

            self.contig_map = ContigMap(self.target_struct, contig)
            self.mappings = self.contig_map.get_mappings()
            self.mask_seq = self.diffusion_mask
            self.mask_str = self.diffusion_mask
            L_mapped = len(self.contig_map.ref)

        ############################
        ### Specific Contig mode ###
        ############################

        else:
            # get contigmap from command line
            assert (
                self.target is None
            ), "Giving a target is the wrong way of handling this is you're doing contigs and secondary structure"

            # process target and reinitialise potential_manager. This is here because the 'target' is always set up to be the second chain in out inputs.
            self.target_feats = iu.process_target(self.inf_conf.input_structure)
            self.contig_map = self.construct_contig(self.target_feats)
            self.mappings = self.contig_map.get_mappings()
            self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None, :]
            self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None, :]
            self.binderlen = len(self.contig_map.inpaint)
            target_feats = self.target_feats
            contig_map = self.contig_map

            xyz_27 = target_feats["xyz_27"]
            mask_27 = target_feats["mask_27"]
            seq_orig = target_feats["seq"]
            L_mapped = len(self.contig_map.ref)
            seq_T = torch.full((L_mapped,), 21)
            seq_T[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
            seq_T[~self.mask_seq.squeeze()] = 21
            assert L_mapped == self.adj.shape[0]
            diffusion_mask = self.mask_str
            self.diffusion_mask = diffusion_mask

            xT = torch.full((1, 1, L_mapped, 27, 3), np.nan)
            xT[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0, ...]
            xT = get_init_xyz(xT).squeeze()
            atom_mask = torch.full((L_mapped, 27), False)
            atom_mask[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

        ####################
        ### Get hotspots ###
        ####################
        self.hotspot_0idx = iu.get_idx0_hotspots(
            self.mappings, self.ppi_conf, self.binderlen
        )

        #########################
        ### Set up potentials ###
        #########################

        self.potential_manager = PotentialManager(
            self.potential_conf,
            self.ppi_conf,
            self.diffuser_conf,
            self.inf_conf,
            self.hotspot_0idx,
            self.binderlen,
        )

        self.chain_idx = ["A" if i < self.binderlen else "B" for i in range(self.L)]

        ########################
        ### Handle Partial T ###
        ########################

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input + 1)
        seq_T = torch.nn.functional.one_hot(seq_T, num_classes=22).float()

        fa_stack, xyz_true = self.diffuser.diffuse_pose(
            xT,
            torch.clone(seq_T),
            atom_mask.squeeze(),
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
        )

        #######################
        ### Set up Denoiser ###
        #######################

        self.denoiser = self.construct_denoiser(self.L, visible=self.mask_seq.squeeze())

        xT = torch.clone(fa_stack[-1].squeeze()[:, :14, :])
        return xT, seq_T

    def _preprocess(self, seq, xyz_t, t):
        (
            msa_masked,
            msa_full,
            seq,
            xyz_prev,
            idx_pdb,
            t1d,
            t2d,
            xyz_t,
            alpha_t,
        ) = super()._preprocess(seq, xyz_t, t)

        ###################################
        ### Add Adj/Secondary Structure ###
        ###################################

        assert (
            self.preprocess_conf.d_t1d == 28
        ), "The checkpoint you're using hasn't been trained with sec-struc/block adjacency features"
        assert (
            self.preprocess_conf.d_t2d == 47
        ), "The checkpoint you're using hasn't been trained with sec-struc/block adjacency features"

        #####################
        ### Handle Target ###
        #####################

        if self.target:
            blank_ss = torch.nn.functional.one_hot(
                torch.full((self.L - self.binderlen,), 3), num_classes=4
            )
            full_ss = torch.cat((self.ss, blank_ss), dim=0)
            if self.scaffold_conf.target_ss is not None:
                full_ss[self.binderlen :] = self.target_ss
        else:
            full_ss = self.ss
        t1d = torch.cat((t1d, full_ss[None, None].to(self.device)), dim=-1)

        t1d = t1d.float()

        ###########
        ### t2d ###
        ###########

        if self.d_t2d == 47:
            if self.target:
                full_adj = torch.zeros((self.L, self.L, 3))
                full_adj[:, :, -1] = 1.0  # set to mask
                full_adj[: self.binderlen, : self.binderlen] = self.adj
                if self.scaffold_conf.target_adj is not None:
                    full_adj[self.binderlen :, self.binderlen :] = self.target_adj
            else:
                full_adj = self.adj
            t2d = torch.cat((t2d, full_adj[None, None].to(self.device)), dim=-1)

        ###########
        ### idx ###
        ###########

        if self.target:
            idx_pdb[:, self.binderlen :] += 200

        return msa_masked, msa_full, seq, xyz_prev, idx_pdb, t1d, t2d, xyz_t, alpha_t
