import math
from typing import List

import numpy as np
import prtm.models.protein_seq_des.data as data
import prtm.models.protein_seq_des.pyrosetta_util as putil
import prtm.models.protein_seq_des.resfile_util as resfile_util
import prtm.models.protein_seq_des.sampler_util as sampler_util
import torch
from prtm import protein
from prtm.models.protein_seq_des import atoms, config
from pyrosetta.rosetta.core.scoring import automorphic_rmsd
from pyrosetta.rosetta.protocols.denovo_design.filters import (
    ExposedHydrophobicsFilterCreator,
)
from pyrosetta.rosetta.protocols.simple_filters import (
    BuriedUnsatHbondFilterCreator,
    PackStatFilterCreator,
)
from torch.distributions.categorical import Categorical


class Sampler(object):
    def __init__(
        self,
        cfg: config.SamplerConfig,
        structure: protein.Protein14,
        models: List[torch.nn.Module],
        init_model: torch.nn.Module,
    ):
        super(Sampler, self).__init__()
        self.structure = structure
        self.cfg = cfg
        self.models = models
        self.init_model = init_model
        self.no_init_model = cfg.no_init_model

        # Use CUDA if models do
        self.use_cuda = list(self.models[0].named_parameters())[0][
            1
        ].is_cuda  # check if model is cuda

        self.iteration = 0
        assert not (
            self.cfg.ala and self.cfg.val
        ), "only ala or val settings can be on for a given run"
        self.chi_mask = None

        self.accept_prob = 1

        # load fixed idx if applicable
        if cfg.fixed_idx != "":
            self.fixed_idx = sampler_util.get_idx(cfg.fixed_idx)
        else:
            self.fixed_idx = []

        # resfile restrictions handling (see util/resfile_util.py)
        if self.cfg.resfile:
            # get resfile NATRO (used to skip designing/packing at all)
            self.fixed_idx = resfile_util.get_natro(self.cfg.resfile)
            # get resfile commands (used to restrict amino acid probability distribution)
            self.resfile = resfile_util.read_resfile(self.cfg.resfile)
            # get initial resfile sequence (used to initialize the sequence)
            self.init_seq_resfile = self.cfg.resfile[2]

            # the initial sequence must be randomized (avoid running the baseline model)
            if self.init_seq_resfile:
                self.cfg.randomize = False

        # load var idx if applicable
        if cfg.var_idx != "":
            self.var_idx = sampler_util.get_idx(cfg.var_idx)
        else:
            self.var_idx = []

        assert not (
            (len(self.fixed_idx) > 0) and (len(self.var_idx) > 0)
        ), "cannot specify both fixed and variable indices"

        if self.no_init_model:
            assert (
                not self.cfg.repack_only
            ), "baseline model must be used for initializing rotamer repacking"

        if self.cfg.symmetry:
            assert (
                len(self.fixed_idx) == 0
            ), "specifying fixed idx not supported in symmetry model"
            assert (
                len(self.var_idx) == 0
            ), "specifying var idx not supported in symmetry model"

    def init(self):
        """initialize sampler
        - initialize rosetta filters
        - score starting (ground-truth) sequence
        - set up constraints on glycines
        - set up symmetry
        - eval metrics on starting (ground-truth) sequence
        - get blocks for blocked sampling
        """

        # initialize sampler
        self.init_rosetta_filters()
        # score starting (ground-truth) pdb, get gt energies
        self.gt_pose = self.structure.to_rosetta_pose()
        self.gt_seq = self.gt_pose.sequence()
        (
            _,
            self.log_p_per_res,
            self.log_p_mean,
            self.logits,
            self.chi_feat,
            self.gt_chi_angles,
            self.gt_chi_mask,
            self.gt_chi,
        ) = sampler_util.get_energy(
            self.models,
            pose=self.gt_pose,
            return_chi=1,
            include_rotamer_probs=1,
            use_cuda=self.use_cuda,
        )
        self.chi_error = 0
        self.re = putil.score_pose(self.gt_pose)
        self.gt_score_terms = self.gt_pose.energies().residue_total_energies_array()
        self.score_terms = list(self.gt_score_terms.dtype.fields)

        # set no gly indices
        ss = self.gt_pose.secstruct()
        self.no_gly_idx = [i for i in range(len(ss)) if ss[i] != "L"]
        self.n = self.gt_pose.residues.__len__()

        # handle symmetry
        if self.cfg.symmetry:
            if self.cfg.is_tim:
                # handle tim case
                self.n_k = (
                    math.ceil((self.n + 1) / self.cfg.k)
                    if (self.n + 1) % 2 == 0
                    else math.ceil((self.n) / self.cfg.k)
                )
            else:
                self.n_k = self.n // self.cfg.k
                assert (
                    self.n % self.cfg.k == 0
                ), "length of protein must be divisible by k for k-fold symm design"
            idx = [
                [
                    i + j * (self.n_k)
                    for j in range(self.cfg.k)
                    if i + j * (self.n_k) < self.n
                ]
                for i in range(self.n_k)
            ]
            self.symmetry_idx = {}
            for idx_set in idx:
                for i in idx_set:
                    self.symmetry_idx[i] = idx_set

            # updated fixed/var idx to reflect symmetry
            for i in self.fixed_idx:
                assert (
                    i in self.symmetry_idx.keys()
                ), "fixed idx must only be specified for first symmetric unit in symmetry mode (within first n_k residues)"
            for i in self.var_idx:
                assert (
                    i in self.symmetry_idx.keys()
                ), "var idx must only be specified for first symmetric unit in symmetry mode (within first n_k residues)"

        # get gt data -- monitor distance to initial sequence
        # Write the structure to a PDB file
        chain_structures = {"A": self.structure.to_biopdb_structure()}
        (
            self.gt_atom_coords,
            self.gt_atom_data,
            self.gt_residue_bb_index_list,
            res_data,
            self.gt_res_label,
            chis,
        ) = data.get_pdb_data(chain_structures)
        self.eval_metrics(self.gt_pose, self.gt_res_label)

        # get conditionally independent blocks via greedy k-colring of backbone 'graph'
        self.get_blocks()

    def init_seq(self):
        # initialize starting sequence

        # random/poly-alanine/poly-valine initialize sequence, pack rotamers
        self.pose = self.structure.to_rosetta_pose()
        if self.cfg.randomize:
            if (not self.no_init_model) and not (self.cfg.ala or self.cfg.val):
                # get features --> BB only
                (
                    res_label,
                    self.log_p_per_res_temp,
                    self.log_p_mean_temp,
                    self.logits_temp,
                    self.chi_feat_temp,
                    self.chi_angles_temp,
                    self.chi_mask_temp,
                ) = sampler_util.get_energy(
                    [self.init_model],
                    self.pose,
                    bb_only=1,
                    include_rotamer_probs=1,
                    use_cuda=self.use_cuda,
                )

                # set sequence
                if not self.cfg.repack_only:
                    # sample res from logits
                    if not self.cfg.symmetry:
                        res, idx, res_label = self.sample(
                            self.logits_temp, np.arange(len(res_label))
                        )
                    else:
                        res, idx, res_label = self.sample(
                            self.logits_temp, np.arange(self.n_k)
                        )
                    # mutate pose residues based on baseline prediction
                    self.pose = putil.mutate_list(
                        self.pose,
                        idx,
                        res,
                        pack_radius=0,
                        fixed_idx=self.fixed_idx,
                        var_idx=self.var_idx,
                    )
                else:
                    res = [i for i in self.gt_seq]
                    if self.cfg.symmetry:
                        res_label = res_label[: self.n_k]

                # sample and set rotamers
                if self.cfg.symmetry:
                    if not self.cfg.repack_only:
                        (
                            self.chi_1,
                            self.chi_2,
                            self.chi_3,
                            self.chi_4,
                            idx,
                            res_idx,
                        ) = self.sample_rotamer(
                            np.arange(self.n_k),
                            [
                                res_label[i]
                                for i in range(0, len(res_label), self.cfg.k)
                            ],
                            self.chi_feat_temp,
                            bb_only=1,
                        )
                    else:
                        (
                            self.chi_1,
                            self.chi_2,
                            self.chi_3,
                            self.chi_4,
                            idx,
                            res_idx,
                        ) = self.sample_rotamer(
                            np.arange(self.n_k),
                            res_label,
                            self.chi_feat_temp,
                            bb_only=1,
                        )
                else:
                    (
                        self.chi_1,
                        self.chi_2,
                        self.chi_3,
                        self.chi_4,
                        idx,
                        res_idx,
                    ) = self.sample_rotamer(
                        np.arange(len(res_label)),
                        res_label,
                        self.chi_feat_temp,
                        bb_only=1,
                    )
                res = [atoms.label_res_single_dict[k] for k in res_idx]
                self.pose = self.set_rotamer(
                    self.pose,
                    res,
                    idx,
                    self.chi_1,
                    self.chi_2,
                    self.chi_3,
                    self.chi_4,
                    fixed_idx=self.fixed_idx,
                    var_idx=self.var_idx,
                )

            # Randomize sequence/rotamers
            else:
                if not self.cfg.repack_only:
                    random_seq = np.random.choice(20, size=len(self.pose))
                    if not self.cfg.ala and not self.cfg.val and self.cfg.symmetry:
                        # random sequence must be symmetric
                        random_seq = np.concatenate(
                            [random_seq[: self.n_k] for i in range(self.cfg.k)]
                        )
                        random_seq = random_seq[: len(self.pose)]
                    self.pose, _ = putil.randomize_sequence(
                        random_seq,
                        self.pose,
                        pack_radius=self.cfg.pack_radius,
                        ala=self.cfg.ala,
                        val=self.cfg.val,
                        resfile_init_seq=self.init_seq_resfile,
                        fixed_idx=self.fixed_idx,
                        var_idx=self.var_idx,
                        repack_rotamers=1,
                    )
                else:
                    assert (
                        False
                    ), "baseline model must be used for initializing rotamer repacking"

        # evaluate energy for starting structure/sequence
        (
            self.res_label,
            self.log_p_per_res,
            self.log_p_mean,
            self.logits,
            self.chi_feat,
            self.chi_angles,
            self.chi_mask,
        ) = sampler_util.get_energy(
            self.models,
            self.pose,
            include_rotamer_probs=1,
            use_cuda=self.use_cuda,
        )
        if self.cfg.repack_only:
            assert np.all(
                self.chi_mask == self.gt_chi_mask
            ), "gt and current pose chi masks should be the same when doing rotamer repacking"

    def init_rosetta_filters(self):
        # initialize pyrosetta filters
        hbond_filter_creator = BuriedUnsatHbondFilterCreator()
        hydro_filter_creator = ExposedHydrophobicsFilterCreator()
        ps_filter_creator = PackStatFilterCreator()
        self.packstat_filter = ps_filter_creator.create_filter()
        self.exposed_hydrophobics_filter = hydro_filter_creator.create_filter()
        self.sc_buried_unsats_filter = hbond_filter_creator.create_filter()
        self.bb_buried_unsats_filter = hbond_filter_creator.create_filter()
        self.bb_buried_unsats_filter.set_report_bb_heavy_atom_unsats(True)
        self.sc_buried_unsats_filter.set_report_sc_heavy_atom_unsats(True)
        self.filters = [
            ("packstat", self.packstat_filter),
            ("exposed_hydrophobics", self.exposed_hydrophobics_filter),
            ("sc_buried_unsats", self.sc_buried_unsats_filter),
            ("bb_buried_unsats", self.bb_buried_unsats_filter),
        ]

    def get_blocks(self, single_res=False):
        # get node blocks for blocked sampling
        D = sampler_util.get_CB_distance(
            self.gt_atom_coords, self.gt_residue_bb_index_list
        )
        if single_res:  # no blocked gibbs -- sampling one res at a time
            self.blocks = [[i] for i in np.arange(D.shape[0])]
            self.n_blocks = len(self.blocks)
        else:
            A = sampler_util.get_graph_from_D(D, self.cfg.threshold)
            # if symmetry holding --> collapse graph st. all neighbors of node i are neighbors of node i+n//4
            if self.cfg.symmetry:
                for i in range(self.n_k):  # //self.cfg.k): #self.graph.shape[0]):
                    A[i] = np.sum(
                        np.concatenate(
                            [
                                A[i + j * self.n_k][None]
                                for j in range(self.cfg.k)
                                if i + j * self.n_k < self.n
                            ]
                        ),
                        axis=0,
                    )
                for i in range(self.n_k):
                    A[:, i] = np.sum(
                        np.concatenate(
                            [
                                A[:, i + j * self.n_k][None]
                                for j in range(self.cfg.k)
                                if i + j * self.n_k < self.n
                            ]
                        ),
                        axis=0,
                    )
                A[A > 1] = 1
                A = A[: self.n_k, : self.n_k]

            self.graph = {i: np.where(A[i, :] == 1)[0] for i in range(A.shape[0])}
            # min k-color of graph by greedy search
            nodes = np.arange(A.shape[0])
            np.random.shuffle(nodes)
            # eliminate fixed indices from list
            if self.cfg.symmetry:
                nodes = [n for n in range(self.n_k)]
            if len(self.fixed_idx) > 0:
                nodes = [n for n in nodes if n not in self.fixed_idx]
            elif len(self.var_idx) > 0:
                nodes = [n for n in nodes if n in self.var_idx]
            self.colors = sampler_util.color_nodes(self.graph, nodes)
            self.n_blocks = 0
            if (
                self.colors
            ):  # check if there are any colored notes to get n-blocks (might be empty if running NATRO on all residues in resfile)
                self.n_blocks = sorted(list(set(self.colors.values())))[-1] + 1
            self.blocks = {}
            for k in self.colors.keys():
                if self.colors[k] not in self.blocks.keys():
                    self.blocks[self.colors[k]] = []
                self.blocks[self.colors[k]].append(k)

        self.reset_block_rate = self.n_blocks

    def eval_metrics(self, pose, res_label):
        self.rosetta_energy = putil.score_pose(pose)
        self.curr_score_terms = pose.energies().residue_total_energies_array()
        self.seq_overlap = (res_label == self.gt_res_label).sum()
        self.filter_scores = []
        for n, filter in self.filters:
            self.filter_scores.append((n, filter.score(pose)))
        if self.cfg.repack_only:
            self.chi_rmsd = sum(
                [
                    automorphic_rmsd(
                        self.gt_pose.residue(i + 1), pose.residue(i + 1), True
                    )
                    for i in range(len(pose))
                ]
            ) / len(pose)
        else:
            self.chi_rmsd = 0
        self.seq = pose.sequence()
        if self.chi_mask is not None and self.cfg.repack_only:
            chi_error = self.chi_mask * np.sqrt(
                (np.sin(self.chi_angles) - np.sin(self.gt_chi_angles)) ** 2
                + (np.cos(self.chi_angles) - np.cos(self.gt_chi_angles)) ** 2
            )
            self.chi_error = np.sum(chi_error) / np.sum(self.chi_mask)
        else:
            self.chi_error = 0

    def enforce_resfile(self, logits, idx):
        """
        enforces resfile constraints by setting logits to -np.inf (see PyTorch on Categorical distribution - returns normalized value)

        logits - tensor where the columns are residue ids, rows are amino acid probabilities
        idx - residue ids
        """
        constraints, header = self.cfg.resfile[0], self.cfg.resfile[1]
        # iterate over all residues and check if they're to be constrained
        for i in idx:
            if i in constraints.keys():
                # set of amino acids to restrict in the tensor
                aa_to_restrict = constraints[i]
                for aa in aa_to_restrict:
                    logits[i, atoms.aa_map_inv[aa]] = -99999
            elif (
                header
            ):  # if not in the constraints, apply header (see util/resfile_util.py)
                aa_to_restrict = header["DEFAULT"]
                for aa in aa_to_restrict:
                    logits[i, atoms.aa_map_inv[aa]] = -99999
        return logits

    def enforce_constraints(self, logits, idx):
        if self.cfg.resfile:
            logits = self.enforce_resfile(logits, idx)
        # enforce idx-wise constraints
        if self.cfg.no_cys:
            logits = logits[..., :-1]
        no_gly_idx = [i for i in idx if i in self.no_gly_idx]
        # note -- definitely other more careful ways to enforce met/gly constraints
        for i in idx:
            if self.cfg.restrict_gly:
                if i in self.no_gly_idx:
                    logits[i, 18] = torch.min(logits[i])
            if self.cfg.no_met:
                logits[i, 13] = torch.min(logits[i])
        if self.cfg.symmetry:
            # average logits across all symmetry postions
            for i in idx:
                logits[i] = torch.cat(
                    [logits[j][None] for j in self.symmetry_idx[i] if j < self.n], 0
                ).mean(0)
        return logits

    def sample_rotamer(self, idx, res_idx, feat, bb_only=0):
        # idx --> (block) residue indices (on chain)
        # res_idx --> idx of residue *type* (AA type)
        # feat --> initial env features from conv net
        assert len(idx) == len(res_idx), (len(idx), len(res_idx))
        if bb_only:
            curr_models = [self.init_model]
        else:
            curr_models = self.models

        if not self.cfg.symmetry:
            # get residue onehot vector
            res_idx_long = torch.LongTensor(res_idx)
            res_onehot = sampler_util.make_onehot(
                res_idx_long.size()[0],
                20,
                res_idx_long[:, None],
                use_cuda=self.use_cuda,
            )

            # get chi feat
            chi_feat = sampler_util.get_chi_init_feat(
                curr_models, feat[idx], res_onehot
            )
            # predict and sample chi angles
            chi_1_pred_out = sampler_util.get_chi_1_logits(curr_models, chi_feat)
            chi_1, chi_1_real, chi_1_onehot = sampler_util.sample_chi(
                chi_1_pred_out, use_cuda=self.use_cuda
            )
            chi_2_pred_out = sampler_util.get_chi_2_logits(
                curr_models, chi_feat, chi_1_onehot
            )
            chi_2, chi_2_real, chi_2_onehot = sampler_util.sample_chi(
                chi_2_pred_out, use_cuda=self.use_cuda
            )
            chi_3_pred_out = sampler_util.get_chi_3_logits(
                curr_models, chi_feat, chi_1_onehot, chi_2_onehot
            )
            chi_3, chi_3_real, chi_3_onehot = sampler_util.sample_chi(
                chi_3_pred_out, use_cuda=self.use_cuda
            )
            chi_4_pred_out = sampler_util.get_chi_4_logits(
                curr_models, chi_feat, chi_1_onehot, chi_2_onehot, chi_3_onehot
            )
            chi_4, chi_4_real, chi_4_onehot = sampler_util.sample_chi(
                chi_4_pred_out, use_cuda=self.use_cuda
            )

            return chi_1_real, chi_2_real, chi_3_real, chi_4_real, idx, res_idx

        else:
            # symmetric rotamer sampling

            # get symmetry indices
            symm_idx = []
            for i in idx:
                symm_idx.extend([j for j in self.symmetry_idx[i]])

            res_idx_symm = []
            for i, idx_i in enumerate(idx):
                res_idx_symm.extend([res_idx[i] for j in self.symmetry_idx[idx_i]])

            # get residue onehot vector
            res_idx_long = torch.LongTensor(res_idx_symm)
            res_onehot = sampler_util.make_onehot(
                res_idx_long.size()[0],
                20,
                res_idx_long[:, None],
                use_cuda=self.use_cuda,
            )

            symm_idx_ptr = []
            count = 0
            for i, idx_i in enumerate(idx):
                symm_idx_ptr.append(
                    [count + j for j in range(len(self.symmetry_idx[idx_i]))]
                )
                count = count + len(self.symmetry_idx[idx_i])

            # get chi feature vector
            chi_feat = sampler_util.get_chi_init_feat(
                curr_models, feat[symm_idx], res_onehot
            )

            # predict and sample chi for each symmetry position
            chi_1_pred_out = sampler_util.get_chi_1_logits(curr_models, chi_feat)
            chi_1_real, chi_1_onehot = sampler_util.get_symm_chi(
                chi_1_pred_out, symm_idx_ptr, use_cuda=self.use_cuda
            )

            chi_2_pred_out = sampler_util.get_chi_2_logits(
                curr_models, chi_feat, chi_1_onehot
            )
            # set debug=True below to reproduce biorxiv results. Sample uniformly 2x from predicted rotamer bin. Small bug for TIM-barrel symmetry experiments for chi_2.
            chi_2_real, chi_2_onehot = sampler_util.get_symm_chi(
                chi_2_pred_out, symm_idx_ptr, use_cuda=self.use_cuda, debug=True
            )

            chi_3_pred_out = sampler_util.get_chi_3_logits(
                curr_models, chi_feat, chi_1_onehot, chi_2_onehot
            )
            chi_3_real, chi_3_onehot = sampler_util.get_symm_chi(
                chi_3_pred_out, symm_idx_ptr, use_cuda=self.use_cuda
            )

            chi_4_pred_out = sampler_util.get_chi_4_logits(
                curr_models, chi_feat, chi_1_onehot, chi_2_onehot, chi_3_onehot
            )
            chi_4_real, chi_4_onehot = sampler_util.get_symm_chi(
                chi_4_pred_out, symm_idx_ptr, use_cuda=self.use_cuda
            )

            return (
                chi_1_real,
                chi_2_real,
                chi_3_real,
                chi_4_real,
                symm_idx,
                res_idx_symm,
            )

    def set_rotamer(
        self, pose, res, idx, chi_1, chi_2, chi_3, chi_4, fixed_idx=[], var_idx=[]
    ):
        # res -- residue type ID
        # idx -- residue index on BB (0-indexed)
        assert len(res) == len(idx)
        assert len(idx) == len(chi_1), (len(idx), len(chi_1))
        for i, r_idx in enumerate(idx):
            if len(fixed_idx) > 0 and r_idx in fixed_idx:
                continue
            elif len(var_idx) > 0 and r_idx not in var_idx:
                continue
            res_i = res[i]
            chi_i = atoms.chi_dict[atoms.aa_inv[res_i]]
            if "chi_1" in chi_i.keys():
                pose.set_chi(1, r_idx + 1, chi_1[i] * (180 / np.pi))
                assert (
                    np.abs(pose.chi(1, r_idx + 1) - chi_1[i] * (180 / np.pi)) <= 1e-5
                ), (pose.chi(1, r_idx + 1), chi_1[i] * (180 / np.pi))
            if "chi_2" in chi_i.keys():
                pose.set_chi(2, r_idx + 1, chi_2[i] * (180 / np.pi))
                assert (
                    np.abs(pose.chi(2, r_idx + 1) - chi_2[i] * (180 / np.pi)) <= 1e-5
                ), (pose.chi(2, r_idx + 1), chi_2[i] * (180 / np.pi))
            if "chi_3" in chi_i.keys():
                pose.set_chi(3, r_idx + 1, chi_3[i] * (180 / np.pi))
                assert (
                    np.abs(pose.chi(3, r_idx + 1) - chi_3[i] * (180 / np.pi)) <= 1e-5
                ), (pose.chi(3, r_idx + 1), chi_3[i] * (180 / np.pi))
            if "chi_4" in chi_i.keys():
                pose.set_chi(4, r_idx + 1, chi_4[i] * (180 / np.pi))
                assert (
                    np.abs(pose.chi(4, r_idx + 1) - chi_4[i] * (180 / np.pi)) <= 1e-5
                ), (pose.chi(4, r_idx + 1), chi_4[i] * (180 / np.pi))

        return pose

    def sample(self, logits, idx):
        # sample residue from model conditional prob distribution at idx with current logits
        logits = self.enforce_constraints(logits, idx)
        dist = Categorical(logits=logits[idx])
        res_idx = dist.sample().cpu().data.numpy()
        idx_out = []
        res = []
        assert len(res_idx) == len(idx), (len(idx), len(res_idx))

        for k in list(res_idx):
            res.append(atoms.label_res_single_dict[k])

        if self.cfg.symmetry:
            idx_out = []
            for i in idx:
                idx_out.extend([j for j in self.symmetry_idx[i] if j < self.n])
            res_out = []
            for i, idx_i in enumerate(idx):
                res_out.extend([res[i] for j in self.symmetry_idx[idx_i] if j < self.n])
            res_idx_out = []
            for i, idx_i in enumerate(idx):
                res_idx_out.extend(
                    [res_idx[i] for j in self.symmetry_idx[idx_i] if j < self.n]
                )

            assert len(idx_out) == len(res_out), (len(idx_out), len(res_out))
            assert len(idx_out) == len(res_idx_out), (len(idx_out), len(res_idx_out))

            return res_out, idx_out, res_idx_out

        return res, idx, res_idx

    def sim_anneal_step(self, e, e_old):
        delta_e = e - e_old
        if delta_e < 0:
            accept_prob = 1.0
        else:
            if self.cfg.anneal_start_temp == 0:
                accept_prob = 0
            else:
                accept_prob = torch.exp(-(delta_e) / self.cfg.anneal_start_temp).item()
        return accept_prob

    def step_T(self):
        # anneal temperature
        self.cfg.anneal_start_temp = max(
            self.cfg.anneal_start_temp * self.cfg.step_rate, self.cfg.anneal_final_temp
        )

    def step(self):
        # no blocks to sample (NATRO for all residues)
        if self.n_blocks == 0:
            self.step_anneal()
            return

        # random idx selection, draw sample
        idx = self.blocks[np.random.choice(self.n_blocks)]

        if not self.cfg.repack_only:
            # sample new residue indices/ residues
            res, idx, res_idx = self.sample(self.logits, idx)
        else:
            # residue idx is fixed (identity fixed) for rotamer repacking
            res = [self.gt_seq[i] for i in idx]
            res_idx = [atoms.aa_map_inv[self.gt_seq[i]] for i in idx]

        # sample rotamer using precomputed chi_feat vector
        (
            self.chi_1,
            self.chi_2,
            self.chi_3,
            self.chi_4,
            idx,
            res_idx,
        ) = self.sample_rotamer(idx, res_idx, self.chi_feat)

        # mutate residues, set rotamers
        res = [atoms.label_res_single_dict[k] for k in res_idx]

        if not self.cfg.use_rosetta_packer:
            # mutate center residue
            if not self.cfg.repack_only:
                self.pose_temp = putil.mutate_list(
                    self.pose,
                    idx,
                    res,
                    pack_radius=0,
                    fixed_idx=self.fixed_idx,
                    var_idx=self.var_idx,
                )

            else:
                self.pose_temp = self.pose

            # sample and set center residue rotamer
            self.pose_temp = self.set_rotamer(
                self.pose_temp,
                res,
                idx,
                self.chi_1,
                self.chi_2,
                self.chi_3,
                self.chi_4,
                fixed_idx=self.fixed_idx,
                var_idx=self.var_idx,
            )

        else:
            # Pyrosetta mutate and rotamer repacking
            self.pose_temp = putil.mutate_list(
                self.pose,
                idx,
                res,
                pack_radius=self.cfg.pack_radius,
                fixed_idx=self.fixed_idx,
                var_idx=self.var_idx,
                repack_rotamers=1,
            )

        # get log prob under model
        (
            self.res_label_temp,
            self.log_p_per_res_temp,
            self.log_p_mean_temp,
            self.logits_temp,
            self.chi_feat_temp,
            self.chi_angles_temp,
            self.chi_mask_temp,
        ) = sampler_util.get_energy(
            self.models,
            self.pose_temp,
            include_rotamer_probs=1,
            use_cuda=self.use_cuda,
        )
        if self.cfg.anneal:
            # simulated annealing accept/reject step
            self.accept_prob = self.sim_anneal_step(
                self.log_p_mean_temp, self.log_p_mean
            )
            r = np.random.uniform(0, 1)
        else:
            # vanilla sampling step
            self.accept_prob = 1
            r = 0

        if r < self.accept_prob:
            # update pose
            self.pose = self.pose_temp
            (
                self.log_p_mean,
                self.log_p_per_res,
                self.logits,
                self.chi_feat,
                self.res_label,
            ) = (
                self.log_p_mean_temp,
                self.log_p_per_res_temp,
                self.logits_temp,
                self.chi_feat_temp,
                self.res_label_temp,
            )
            self.chi_angles, self.chi_mask = self.chi_angles_temp, self.chi_mask_temp

            # eval all metrics
            self.eval_metrics(self.pose, self.res_label)

            self.step_anneal()

    def step_anneal(self):
        # ending for step()
        if self.cfg.anneal:
            self.step_T()

        self.iteration += 1

        # reset blocks
        if self.reset_block_rate != 0 and (self.iteration % self.reset_block_rate == 0):
            self.get_blocks()