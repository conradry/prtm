from dataclasses import asdict

import torch
import torch.nn as nn
from proteome.models.folding.rosettafold.attention_module import \
    IterativeFeatureExtractor
from proteome.models.folding.rosettafold.distance_predictor import \
    DistanceNetwork
from proteome.models.folding.rosettafold.embeddings import (MSA_emb,
                                                            Pair_emb_w_templ,
                                                            Pair_emb_wo_templ,
                                                            Templ_emb)
from proteome.models.folding.rosettafold.refine_module import Refine_module
from proteome.models.folding.rosettafold.config import RoseTTAFoldConfig


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
            self.pair_emb = Pair_emb_wo_templ(d_model=config.d_pair, p_drop=config.p_drop)
 
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

    def forward(
        self,
        msa,
        seq,
        idx,
        t1d=None,
        t2d=None,
        prob_s=None,
        return_raw=False,
        refine_only=False,
    ):
        seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()
        B, N, L = msa.shape
        if refine_only:
            ref_xyz, ref_lddt = self.refine(msa, prob_s, seq1hot, idx)
            return ref_xyz, ref_lddt.view(B, L)
        else:
            # Get embeddings
            msa = self.msa_emb(msa, idx)
            if self.use_templ:
                tmpl = self.templ_emb(t1d, t2d, idx)
                pair = self.pair_emb(seq, idx, tmpl)
            else:
                pair = self.pair_emb(seq, idx)
            #
            # Extract features
            msa, pair, xyz, lddt = self.feat_extractor(msa, pair, seq1hot, idx)

            # Predict 6D coords
            logits = self.c6d_predictor(pair)

            prob_s = list()
            for l in logits:
                prob_s.append(nn.Softmax(dim=1)(l))  # (B, C, L, L)
            prob_s = torch.cat(prob_s, dim=1).permute(0, 2, 3, 1)

            if return_raw:
                return logits, msa, xyz, lddt.view(B, L)
            else:
                ref_xyz, ref_lddt = self.refine(msa, prob_s, seq1hot, idx)
                return ref_xyz, ref_lddt.view(B, L)
