from proteome.models.design.genie.model.pair_feature_net import PairFeatureNet
from proteome.models.design.genie.model.pair_transform_net import \
    PairTransformNet
from proteome.models.design.genie.model.single_feature_net import \
    SingleFeatureNet
from proteome.models.design.genie.model.structure_net import StructureNet
from torch import nn

from proteome.models.design.genie import config


class Denoiser(nn.Module):
    def __init__(
        self, cfg: config.GenieConfig, n_timestep: int
    ):
        super(Denoiser, self).__init__()

        self.single_feature_net = SingleFeatureNet(
            cfg.c_s, n_timestep, cfg.c_pos_emb, cfg.c_timestep_emb
        )

        self.pair_feature_net = PairFeatureNet(cfg.c_s, cfg.c_p, cfg.relpos_k, cfg.template_type)

        self.pair_transform_net = (
            PairTransformNet(
                cfg.c_p,
                cfg.n_pair_transform_layer,
                cfg.include_mul_update,
                cfg.include_tri_att,
                cfg.c_hidden_mul,
                cfg.c_hidden_tri_att,
                cfg.n_head_tri,
                cfg.tri_dropout,
                cfg.pair_transition_n,
            )
            if cfg.n_pair_transform_layer > 0
            else None
        )

        self.structure_net = StructureNet(
            cfg.c_s,
            cfg.c_p,
            cfg.n_structure_layer,
            cfg.n_structure_block,
            cfg.c_hidden_ipa,
            cfg.n_head_ipa,
            cfg.n_qk_point,
            cfg.n_v_point,
            cfg.ipa_dropout,
            cfg.n_structure_transition_layer,
            cfg.structure_transition_dropout,
        )

    def forward(self, ts, timesteps, mask):
        p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        s = self.single_feature_net(ts, timesteps, mask)
        p = self.pair_feature_net(s, ts, p_mask)
        if self.pair_transform_net is not None:
            p = self.pair_transform_net(p, p_mask)
        ts = self.structure_net(s, p, ts, mask)
        return ts
