from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from prtm.models.rosettafold.attention_module import IterativeFeatureExtractor
from prtm.models.rosettafold.config import RoseTTAFoldConfig
from prtm.models.rosettafold.distance_predictor import DistanceNetwork
from prtm.models.rosettafold.embeddings import (
    MSA_emb,
    Pair_emb_w_templ,
    Pair_emb_wo_templ,
    Templ_emb,
)
from prtm.models.rosettafold.refine_module import Refine_module


class RoseTTAFold(nn.Module):
    """This is the end-to-end RoseTTAFold model"""

    def __init__(self, config: RoseTTAFoldConfig):
        super(RoseTTAFold, self).__init__()
        self.use_templ = config.use_templ

        self.msa_emb = MSA_emb(d_model=config.d_msa, p_drop=config.p_drop, max_len=5000)
        if config.use_templ:
            self.templ_emb = Templ_emb(
                d_templ=config.d_templ,
                n_att_head=config.n_head_templ,
                r_ff=config.r_ff,
                performer_opts=asdict(config.performer_L_opts),
                p_drop=0.0,
            )
            self.pair_emb = Pair_emb_w_templ(
                d_model=config.d_pair, d_templ=config.d_templ, p_drop=config.p_drop
            )
        else:
            self.pair_emb = Pair_emb_wo_templ(
                d_model=config.d_pair, p_drop=config.p_drop
            )

        self.feat_extractor = IterativeFeatureExtractor(
            n_module=config.n_module,
            n_module_str=config.n_module_str,
            n_layer=config.n_layer,
            d_msa=config.d_msa,
            d_pair=config.d_pair,
            d_hidden=config.d_hidden,
            n_head_msa=config.n_head_msa,
            n_head_pair=config.n_head_pair,
            r_ff=config.r_ff,
            n_resblock=config.n_resblock,
            p_drop=config.p_drop,
            performer_N_opts=asdict(config.performer_N_opts),
            performer_L_opts=asdict(config.performer_L_opts),
            SE3_param=asdict(config.se3_config),
        )
        self.c6d_predictor = DistanceNetwork(config.d_pair, p_drop=config.p_drop)

        self.refine = Refine_module(
            config.n_module_ref,
            d_node=config.d_msa,
            d_pair=130,
            d_node_hidden=config.d_hidden,
            d_pair_hidden=config.d_hidden,
            SE3_param=asdict(config.refinement_config),
            p_drop=config.p_drop,
        )

    def run_refinement(self, msa, seq, idx, prob_s=None):
        """Runs only the refinement module."""
        B, N, L = msa.shape
        seq1hot = F.one_hot(seq, num_classes=21).float()
        ref_xyz, ref_lddt = self.refine(msa, prob_s, seq1hot, idx)
        return ref_xyz, ref_lddt.view(B, L)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        refine: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """Runs the full model."""
        msa = batch["msa"]
        seq = batch["seq"]
        idx = batch["idx"]
        t1d = batch["t1d"]
        t2d = batch["t2d"]

        seq1hot = F.one_hot(seq, num_classes=21).float()
        B, N, L = msa.shape
        # Get embeddings
        msa = self.msa_emb(msa, idx)
        if self.use_templ:
            tmpl = self.templ_emb(t1d, t2d, idx)
            pair = self.pair_emb(seq, idx, tmpl)
        else:
            pair = self.pair_emb(seq, idx)

        # Extract features
        msa, pair, xyz, lddt = self.feat_extractor(msa, pair, seq1hot, idx)

        # Predict 6D coords
        logits = self.c6d_predictor(pair)
        prob_s = tuple([F.softmax(l, dim=1) for l in logits])
        if refine:
            prob_s_refine = torch.cat(prob_s, dim=1)
            prob_s_refine = prob_s_refine.permute(0, 2, 3, 1)
            xyz, lddt = self.refine(msa, prob_s_refine, seq1hot, idx)

        return prob_s, xyz, lddt.view(B, L)
