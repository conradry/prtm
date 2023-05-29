import torch
import torch.nn as nn
import torch.nn.functional as F
from proteome.models.folding.rosettafold.rosetta_transformer import Encoder, EncoderLayer
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from proteome.models.folding.rosettafold.rosetta_transformer import SequenceWeight


def get_seqsep(idx, clamp=False):
    """
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
                  Sergey found that having sign in seqsep features helps a little
    """
    seqsep = idx[:, None, :] - idx[:, :, None]
    sign = torch.sign(seqsep)
    seqsep = torch.log(torch.abs(seqsep) + 1.0)
    if clamp:
        seqsep = torch.clamp(seqsep, 0.0, 5.5)
 
    seqsep = sign * seqsep
    return seqsep.unsqueeze(-1)


def make_graph(node, idx, emb):
    """create torch_geometric graph from Trunk outputs"""
    device = emb.device
    B, L = emb.shape[:2]

    # |i-j| <= kmin (connect sequentially adjacent residues)
    sep = idx[:, None, :] - idx[:, :, None]
    sep = sep.abs()
    b, i, j = torch.where(sep > 0)

    src = b * L + i
    tgt = b * L + j

    x = node.reshape(B * L, -1)

    G = Data(x=x, edge_index=torch.stack([src, tgt]), edge_attr=emb[b, i, j])

    return G


def get_tiled_1d_features(seq, node=None):
    """
    Input:
        - seq: target sequence in integer (B,L)
    Output:
        - tiled 1d features including 1hot encoded sequence (B, L, 21) and node features if given
    """
    B, L = seq.shape
    #
    seq1hot = F.one_hot(seq, num_classes=21).float()
    if node != None:
        feat_1d = torch.cat((seq1hot, node), dim=-1)
    else:
        feat_1d = seq1hot

    left = feat_1d.view(B, L, 1, -1).expand(-1, -1, L, -1)
    right = feat_1d.view(B, 1, L, -1).expand(-1, L, -1, -1)
    return torch.cat((left, right), dim=-1)  # (B, L, L, -1)


class Attention(nn.Module):
    def __init__(self, d_model=128, d_attn=50):
        super(Attention, self).__init__()
        #
        self.to_v = nn.Linear(d_model, d_attn)
        self.to_u = nn.Linear(d_attn, 1, bias=False)

    def forward(self, x, time_major=False):
        if time_major:
            L, BL = x.shape[:2]
            x = x.permute(1, 0, 2)  # make it as (Batch, Time, Feats)
        else:
            BL, L = x.shape[:2]
        v = torch.tanh(self.to_v(x))  # (B, T, A)
        vu = self.to_u(v).view(BL, L)  # (B, T)
        alphas = F.softmax(vu, dim=-1).view(BL, L, 1)

        x = torch.matmul(alphas.transpose(-2, -1), x).view(BL, -1)

        return x


class UniMPBlock(nn.Module):
    """https://arxiv.org/pdf/2009.03509.pdf"""

    def __init__(self, node_dim=64, edge_dim=64, heads=4, dropout=0.15):
        super(UniMPBlock, self).__init__()

        self.TConv = TransformerConv(
            node_dim, node_dim, heads, dropout=dropout, edge_dim=edge_dim
        )
        self.LNorm = nn.LayerNorm(node_dim * heads)
        self.Linear = nn.Linear(node_dim * heads, node_dim)
        self.Activ = nn.ELU(inplace=True)

    # @torch.cuda.amp.autocast(enabled=True)
    def forward(self, G):
        xin, e_idx, e_attr = G.x, G.edge_index, G.edge_attr
        x = self.TConv(xin, e_idx, e_attr)
        x = self.LNorm(x)
        x = self.Linear(x)
        out = self.Activ(x + xin)
        return Data(x=out, edge_index=e_idx, edge_attr=e_attr)


class InitStr_Network(nn.Module):
    def __init__(
        self,
        node_dim_in=64,
        node_dim_hidden=64,
        edge_dim_in=128,
        edge_dim_hidden=64,
        nheads=4,
        nblocks=3,
        dropout=0.1,
    ):
        super(InitStr_Network, self).__init__()

        # embedding layers for node and edge features
        self.norm_node = nn.LayerNorm(node_dim_in)
        self.norm_edge = nn.LayerNorm(edge_dim_in)
        self.encoder_seq = SequenceWeight(node_dim_in, 1, dropout=dropout)

        self.embed_x = nn.Sequential(
            nn.Linear(node_dim_in + 21, node_dim_hidden), nn.ELU(inplace=True)
        )
        self.embed_e = nn.Sequential(
            nn.Linear(edge_dim_in + 1, edge_dim_hidden), nn.ELU(inplace=True)
        )

        # graph transformer
        blocks = [
            UniMPBlock(node_dim_hidden, edge_dim_hidden, nheads, dropout)
            for _ in range(nblocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        # outputs
        self.get_xyz = nn.Linear(node_dim_hidden, 9)

    def forward(self, seq1hot, idx, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm_node(msa)
        pair = self.norm_edge(pair)

        w_seq = self.encoder_seq(msa).reshape(B, L, 1, N).permute(0, 3, 1, 2)
        msa = w_seq * msa
        msa = msa.sum(dim=1)
        node = torch.cat((msa, seq1hot), dim=-1)
        node = self.embed_x(node)

        seqsep = get_seqsep(idx)
        pair = torch.cat((pair, seqsep), dim=-1)
        pair = self.embed_e(pair)

        G = make_graph(node, idx, pair)
        Gout = self.transformer(G)

        xyz = self.get_xyz(Gout.x)

        return xyz.reshape(B, L, 3, 3)  # torch.cat([xyz,node_emb],dim=-1)

    
class InitStr_Network2Track(nn.Module):
    def __init__(
        self,
        d_model=128,
        d_hidden=64,
        d_out=64,
        d_attn=50,
        d_msa=64,
        n_layers=2,
        n_att_head=4,
        r_ff=2,
        p_drop=0.1,
        performer_opts=None,
    ):
        super(InitStr_Network2Track, self).__init__()
        self.norm_node = nn.LayerNorm(d_msa)
        self.norm_edge = nn.LayerNorm(d_model)
        #
        self.proj_mix = nn.Linear(d_model + d_msa * 2 + 21 * 2 + 1, d_hidden)

        enc_layer_1 = EncoderLayer(
            d_model=d_hidden,
            d_ff=d_hidden * r_ff,
            heads=n_att_head,
            p_drop=p_drop,
            performer_opts=performer_opts,
            is_2track=True,
        )
        self.encoder_1 = Encoder(enc_layer_1, n_layers)

        enc_layer_2 = EncoderLayer(
            d_model=d_hidden,
            d_ff=d_hidden * r_ff,
            heads=n_att_head,
            p_drop=p_drop,
            performer_opts=performer_opts,
            is_2track=True,
        )
        self.encoder_2 = Encoder(enc_layer_2, n_layers)

        self.attn = Attention(d_model=d_hidden * 2, d_attn=d_attn)
        self.proj = nn.Linear(d_hidden * 2, d_out)

        enc_layer_3 = EncoderLayer(
            d_model=d_out,
            d_ff=d_out * r_ff,
            heads=n_att_head,
            p_drop=p_drop,
            performer_opts=performer_opts,
        )
        self.encoder_3 = Encoder(enc_layer_3, n_layers)

        self.proj_crd = nn.Linear(d_out, 9)  # predict BB coordinates

    def forward(self, msa, pair, seq, idx):
        B, L = pair.shape[:2]
        msa = self.norm_node(msa)
        pair = self.norm_edge(pair)
        #
        node_feats = msa.mean(1)  # (B, L, K)
        #
        tiled_1d = get_tiled_1d_features(seq, node=node_feats)
        seqsep = get_seqsep(idx)
        pair = torch.cat((pair, seqsep, tiled_1d), dim=-1)
        pair = self.proj_mix(pair)

        # reduce dimension
        hidden_1 = self.encoder_1(pair).view(B * L, L, -1)  # (B*L, L, d_hidden)
        #
        pair = pair.view(B, L, L, -1).permute(0, 2, 1, 3)
        hidden_2 = self.encoder_2(pair).reshape(B * L, L, -1)  # (B*L, L, d_hidden)
        pair = torch.cat((hidden_1, hidden_2), dim=-1)
        out = self.attn(pair)  # (B*L, d_hidden)
        out = self.proj(out).view(B, L, -1)  # (B, L, d_out)
        #
        out = self.encoder_3(out.reshape(B, 1, L, -1))
        xyz = self.proj_crd(out).view(B, L, 3, 3)  # (B, L, 3, 3)

        return xyz
