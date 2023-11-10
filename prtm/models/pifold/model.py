import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prtm.models.pifold.config import PiFoldConfig
from prtm.models.pifold.simdesign_utils import (
    _dihedrals,
    _get_rbf,
    _orientations_coarse_gl_tuple,
    gather_nodes,
)
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum


def get_attend_mask(idx, mask):
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    return mask_attend


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=True):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp

        self.W_V = nn.Sequential(
            nn.Linear(num_in, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
        )
        self.Bias = nn.Sequential(
            nn.Linear(num_hidden * 3, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_heads),
        )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id, dst_idx=None):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        w = self.Bias(torch.cat([h_V[center_id], h_E], dim=-1)).view(E, n_heads, 1)
        attend_logits = w / np.sqrt(d)

        V = self.W_V(h_E).view(-1, n_heads, d)
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend * V, center_id, dim=0).view([-1, self.num_hidden])

        if self.output_mlp:
            h_V_update = self.W_O(h_V)
        else:
            h_V_update = h_V
        return h_V_update


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EdgeMLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        num_heads=None,
        scale=30,
        node_context=False,
        edge_context=False,
    ):
        super(Context, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context

        self.V_MLP = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
        )

        self.V_MLP_g = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid(),
        )

        self.E_MLP = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
        )

        self.E_MLP_g = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid(),
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        if self.node_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)
            h_V = h_V * self.V_MLP_g(c_V[batch_id])

        if self.edge_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)
            h_E = h_E * self.E_MLP_g(c_V[batch_id[edge_idx[0]]])

        return h_V, h_E


class GeneralGNN(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        num_heads=None,
        scale=30,
        node_net="AttMLP",
        edge_net="EdgeMLP",
        node_context=0,
        edge_context=0,
    ):
        super(GeneralGNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        self.node_net = node_net
        self.edge_net = edge_net
        if node_net == "AttMLP":
            self.attention = NeighborAttention(num_hidden, num_in, num_heads=4)
        if edge_net == "None":
            pass
        if edge_net == "EdgeMLP":
            self.edge_update = EdgeMLP(num_hidden, num_in, num_heads=4)

        self.context = Context(
            num_hidden,
            num_in,
            num_heads=4,
            node_context=node_context,
            edge_context=edge_context,
        )

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden),
        )
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        if self.node_net == "AttMLP" or self.node_net == "QKV":
            dh = self.attention(
                h_V, torch.cat([h_E, h_V[dst_idx]], dim=-1), src_idx, batch_id, dst_idx
            )
        else:
            dh = self.attention(h_V, h_E, src_idx, batch_id, dst_idx)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if self.edge_net == "None":
            pass
        else:
            h_E = self.edge_update(h_V, h_E, edge_idx, batch_id)

        h_V, h_E = self.context(h_V, h_E, edge_idx, batch_id)
        return h_V, h_E


class StructureEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_encoder_layers=3,
        dropout=0,
        node_net="AttMLP",
        edge_net="EdgeMLP",
        node_context=True,
        edge_context=False,
    ):
        """Graph labeling network"""
        super(StructureEncoder, self).__init__()
        encoder_layers = []

        module = GeneralGNN

        for i in range(num_encoder_layers):
            encoder_layers.append(
                module(
                    hidden_dim,
                    hidden_dim * 2,
                    dropout=dropout,
                    node_net=node_net,
                    edge_net=edge_net,
                    node_context=node_context,
                    edge_context=edge_context,
                ),
            )

        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self, h_V, h_P, P_idx, batch_id):
        for layer in self.encoder_layers:
            h_V, h_P = layer(h_V, h_P, P_idx, batch_id)
        return h_V, h_P


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab=20):
        super().__init__()
        self.readout = nn.Linear(hidden_dim, vocab)

    def forward(self, h_V, batch_id=None):
        logits = self.readout(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits


class PiFoldModel(nn.Module):
    def __init__(self, cfg: PiFoldConfig):
        """Graph labeling network"""
        super(PiFoldModel, self).__init__()
        self.cfg = cfg
        node_features = cfg.node_features
        edge_features = cfg.edge_features
        hidden_dim = cfg.hidden_dim
        dropout = cfg.dropout
        num_encoder_layers = cfg.num_encoder_layers
        self.top_k = cfg.k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16
        self.virtual_atoms = nn.Parameter(torch.rand(self.cfg.virtual_num, 3))

        node_in = 0
        if self.cfg.node_dist:
            pair_num = 6
            if self.cfg.virtual_num > 0:
                pair_num += self.cfg.virtual_num * (self.cfg.virtual_num - 1)
            node_in += pair_num * self.num_rbf

        if self.cfg.node_angle:
            node_in += 12
        if self.cfg.node_direct:
            node_in += 9

        edge_in = 0
        if self.cfg.edge_dist:
            pair_num = 16

            if self.cfg.virtual_num > 0:
                pair_num += self.cfg.virtual_num
                pair_num += self.cfg.virtual_num * (self.cfg.virtual_num - 1)

            edge_in += pair_num * self.num_rbf

        if self.cfg.edge_angle:
            edge_in += 4
        if self.cfg.edge_direct:
            edge_in += 12

        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        self.W_v = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout)

        self.decoder = MLPDecoder(hidden_dim)
        self._init_params()

    def forward(
        self,
        h_V,
        h_P,
        P_idx,
        batch_id,
        S=None,
        AT_test=False,
        mask_bw=None,
        mask_fw=None,
        decoding_order=None,
        return_logit=False,
    ):
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))

        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)

        log_probs, logits = self.decoder(h_V, batch_id)

        if return_logit == True:
            return log_probs, logits
        return log_probs

    def _init_params(self):
        for name, p in self.named_parameters():
            if name == "virtual_atoms":
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = (1.0 - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * (D_max + 1)
        D_neighbors, E_idx = torch.topk(
            D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _get_features(self, S, score, X, mask):
        device = X.device
        mask_bool = mask == 1
        B, N, _, _ = X.shape
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(
            x, mask_attend.unsqueeze(-1)
        ).reshape(-1, x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(
            x, mask_bool.unsqueeze(-1)
        ).reshape(-1, x.shape[-1])

        randn = torch.rand(mask.shape, device=X.device) + 5
        decoding_order = torch.argsort(
            -mask * (torch.abs(randn))
        )  # 我们的mask=1代表数据可用, 而protein MPP的mask=1代表数据不可用，正好相反
        mask_size = mask.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        # 计算q已知的情况下, q->p的mask,
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend2 = torch.gather(order_mask_backward, 2, E_idx)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1])
        mask_bw = (mask_1D * mask_attend2).unsqueeze(-1)
        mask_fw = (mask_1D * (1 - mask_attend2)).unsqueeze(-1)
        mask_bw = edge_mask_select(mask_bw).squeeze()
        mask_fw = edge_mask_select(mask_fw).squeeze()

        # sequence
        S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)

        # angle & direction
        V_angles = _dihedrals(X, 0)
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        # distance
        atom_N = X[:, :, 0, :]
        atom_Ca = X[:, :, 1, :]
        atom_C = X[:, :, 2, :]
        atom_O = X[:, :, 3, :]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        if self.cfg.virtual_num > 0:
            virtual_atoms = self.virtual_atoms / torch.norm(
                self.virtual_atoms, dim=1, keepdim=True
            )
            for i in range(self.virtual_atoms.shape[0]):
                vars()["atom_v" + str(i)] = (
                    virtual_atoms[i][0] * a
                    + virtual_atoms[i][1] * b
                    + virtual_atoms[i][2] * c
                    + 1 * atom_Ca
                )

        node_list = ["Ca-N", "Ca-C", "Ca-O", "N-C", "N-O", "O-C"]
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split("-")
            node_dist.append(
                node_mask_select(
                    _get_rbf(
                        vars()["atom_" + atom1],
                        vars()["atom_" + atom2],
                        None,
                        self.num_rbf,
                    ).squeeze()
                )
            )

        if self.cfg.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                # # true atoms
                for j in range(0, i):
                    node_dist.append(
                        node_mask_select(
                            _get_rbf(
                                vars()["atom_v" + str(i)],
                                vars()["atom_v" + str(j)],
                                None,
                                self.num_rbf,
                            ).squeeze()
                        )
                    )
                    node_dist.append(
                        node_mask_select(
                            _get_rbf(
                                vars()["atom_v" + str(j)],
                                vars()["atom_v" + str(i)],
                                None,
                                self.num_rbf,
                            ).squeeze()
                        )
                    )
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()

        pair_lst = [
            "Ca-Ca",
            "Ca-C",
            "C-Ca",
            "Ca-N",
            "N-Ca",
            "Ca-O",
            "O-Ca",
            "C-C",
            "C-N",
            "N-C",
            "C-O",
            "O-C",
            "N-N",
            "N-O",
            "O-N",
            "O-O",
        ]

        edge_dist = []  # Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split("-")
            rbf = _get_rbf(
                vars()["atom_" + atom1], vars()["atom_" + atom2], E_idx, self.num_rbf
            )
            edge_dist.append(edge_mask_select(rbf))

        if self.cfg.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(
                    edge_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(i)],
                            vars()["atom_v" + str(i)],
                            E_idx,
                            self.num_rbf,
                        )
                    )
                )
                for j in range(0, i):
                    edge_dist.append(
                        edge_mask_select(
                            _get_rbf(
                                vars()["atom_v" + str(i)],
                                vars()["atom_v" + str(j)],
                                E_idx,
                                self.num_rbf,
                            )
                        )
                    )
                    edge_dist.append(
                        edge_mask_select(
                            _get_rbf(
                                vars()["atom_v" + str(j)],
                                vars()["atom_v" + str(i)],
                                E_idx,
                                self.num_rbf,
                            )
                        )
                    )

        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        h_V = []
        if self.cfg.node_dist:
            h_V.append(V_dist)
        if self.cfg.node_angle:
            h_V.append(V_angles)
        if self.cfg.node_direct:
            h_V.append(V_direct)

        h_E = []
        if self.cfg.edge_dist:
            h_E.append(E_dist)
        if self.cfg.edge_angle:
            h_E.append(E_angles)
        if self.cfg.edge_direct:
            h_E.append(E_direct)

        _V = torch.cat(h_V, dim=-1)
        _E = torch.cat(h_E, dim=-1)

        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B, 1, 1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1, -1)
        dst = shift.view(B, 1, 1) + torch.arange(0, N, device=src.device).view(
            1, -1, 1
        ).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1, -1)
        E_idx = torch.cat((dst, src), dim=0).long()

        decoding_order = (
            node_mask_select((decoding_order + shift.view(-1, 1)).unsqueeze(-1))
            .squeeze()
            .long()
        )

        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X = X[sparse_idx[:, 0], sparse_idx[:, 1], :, :]
        batch_id = sparse_idx[:, 0]

        return X, S, score, _V, _E, E_idx, batch_id, mask_bw, mask_fw, decoding_order
