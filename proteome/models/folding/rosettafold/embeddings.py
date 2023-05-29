import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from proteome.models.folding.rosettafold.rosetta_transformer import AxialEncoderLayer, Encoder, EncoderLayer

# Initial embeddings for target sequence, msa, template info
# positional encoding
#   option 1: using sin/cos --> using this for now
#   option 2: learn positional embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_drop=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.drop = nn.Dropout(p_drop)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # (1, max_len, d_model)

    def forward(self, x, idx_s):
        pe = list()
        for idx in idx_s:
            pe.append(self.pe[:, idx, :])
        pe = torch.stack(pe)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.drop = nn.Dropout(p_drop, inplace=True)
        #
        d_model_half = d_model // 2
        div_term = torch.exp(
            torch.arange(0.0, d_model_half, 2) * -(math.log(10000.0) / d_model_half)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x, idx_s):
        B, L, _, K = x.shape
        K_half = K // 2
        pe = torch.zeros_like(x)
        i_batch = -1
        for idx in idx_s:
            i_batch += 1
            sin_inp = idx.unsqueeze(1) * self.div_term
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)  # (L, K//2)
            pe[i_batch, :, :, :K_half] = emb.unsqueeze(1)
            pe[i_batch, :, :, K_half:] = emb.unsqueeze(0)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)


class QueryEncoding(nn.Module):
    def __init__(self, d_model):
        super(QueryEncoding, self).__init__()
        self.pe = nn.Embedding(2, d_model)  # (0 for query, 1 for others)

    def forward(self, x):
        B, N, L, K = x.shape
        idx = torch.ones((B, N, L), device=x.device).long()
        idx[:, 0, :] = 0  # first sequence is the query
        x = x + self.pe(idx)
        return x


class MSA_emb(nn.Module):
    def __init__(
        self, 
        d_model=64, 
        d_msa=21, 
        p_drop=0.1, 
        max_len=5000,
        query_encoding=False,
    ):
        super(MSA_emb, self).__init__()
        self.emb = nn.Embedding(d_msa, d_model)
        self.pos = PositionalEncoding(d_model, p_drop=p_drop, max_len=max_len)
        self.query_encoding = query_encoding
        if query_encoding:
            self.pos_q = QueryEncoding(d_model)

    def forward(self, msa, idx):
        B, N, L = msa.shape
        out = self.emb(msa)  # (B, N, L, K//2)
        out = self.pos(out, idx)  # add positional encoding
        if self.query_encoding:
            out = self.pos_q(out)
        return out


# pixel-wise attention based embedding (from trRosetta-tbm)
class Templ_emb(nn.Module):
    def __init__(
        self,
        d_t1d=3,
        d_t2d=10,
        d_templ=64,
        n_att_head=4,
        r_ff=4,
        performer_opts=None,
        p_drop=0.1,
        network_2track=False,
    ):
        super(Templ_emb, self).__init__()
        self.proj = nn.Linear(d_t1d * 2 + d_t2d + 1, d_templ)
        self.network_2track = network_2track
        if not network_2track:
            self.pos = PositionalEncoding2D(d_templ, p_drop=p_drop)
            enc_layer = AxialEncoderLayer(
                d_templ,
                d_templ * r_ff,
                n_att_head,
                p_drop=p_drop,
                performer_opts=performer_opts,
            )
            self.norm = nn.LayerNorm(d_templ)
            self.to_attn = nn.Linear(d_templ, 1)
        else:
            enc_layer = EncoderLayer(
                d_model=d_templ, d_ff=d_templ * r_ff, heads=n_att_head, p_drop=p_drop
            )
            self.to_v = nn.Linear(d_templ, d_templ)
            self.to_u = nn.Linear(d_templ, 1, bias=False)

        self.encoder = Encoder(enc_layer, 1)

    def forward(self, t1d, t2d, idx):
        # Input
        #   - t1d: 1D template info (B, T, L, 2)
        #   - t2d: 2D template info (B, T, L, L, 10)
        B, T, L, _ = t1d.shape
        left = t1d.unsqueeze(3).repeat(1, 1, 1, L, 1)
        right = t1d.unsqueeze(2).repeat(1, 1, L, 1, 1)
        seqsep = torch.abs(idx[:, :, None] - idx[:, None, :]) + 1
        seqsep = (
            torch.log(seqsep.float())
            .view(B, L, L, 1)
            .unsqueeze(1)
            .repeat(1, T, 1, 1, 1)
        )
        
        feat = torch.cat((t2d, left, right, seqsep), -1)
        feat = self.proj(feat)
        if not self.network_2track:
            feat = feat.reshape(B * T, L, L, -1)
            tmp = self.pos(feat, idx)  # add positional embedding
        
            # attention along L
            feat = torch.empty_like(tmp)
            for i_f in range(tmp.shape[0]):
                feat[i_f] = self.encoder(tmp[i_f].view(1, L, L, -1))
            del tmp
            feat = feat.reshape(B, T, L, L, -1)
            feat = feat.permute(0, 2, 3, 1, 4).contiguous().view(B, L * L, T, -1)

            attn = self.to_attn(self.norm(feat))
            attn = F.softmax(attn, dim=-2)  # (B, L*L, T, 1)
            feat = torch.matmul(attn.transpose(-2, -1), feat)

        if self.network_2track:
            feat = feat.permute(0, 2, 3, 1, 4).contiguous().view(B, L * L, T, -1)
            feat = self.encoder(feat).view(B * L * L, T, -1)
            v = torch.tanh(self.to_v(feat))  # (B*L*L, T, A)
            vu = self.to_u(v).view(B * L * L, T)
            alphas = F.softmax(vu, dim=-1).view(B * L * L, T, 1)  # attention map
            feat = torch.matmul(alphas.transpose(-2, -1), feat)

        return feat.reshape(B, L, L, -1).view(B, L, L, -1)


class Pair_emb_w_templ(nn.Module):
    def __init__(
        self, 
        d_model=128, 
        d_seq=21, 
        d_templ=64, 
        p_drop=0.1,
        network_2track=False,
    ):
        super(Pair_emb_w_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.projection = nn.Linear(d_model + d_templ + 1, d_model)
        self.network_2track = network_2track
        if not network_2track:
            self.norm_templ = nn.LayerNorm(d_templ)
            self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)

    def forward(self, seq, idx, templ):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        #
        # get initial sequence pair features
        seq = self.emb(seq)  # (B, L, d_model//2)
        left = seq.unsqueeze(2).repeat(1, 1, L, 1)
        right = seq.unsqueeze(1).repeat(1, L, 1, 1)
        seqsep = torch.abs(idx[:, :, None] - idx[:, None, :]) + 1
        seqsep = torch.log(seqsep.float()).view(B, L, L, 1)
        
        if not self.network_2track:
            templ = self.norm_templ(templ)
            pair = torch.cat((left, right, seqsep, templ), dim=-1)
        else:
            pair = torch.cat((left, right, templ, seqsep), dim=-1)
            
        pair = self.projection(pair)
        if not self.network_2track:
            pair = self.pos(pair, idx)
        return pair


class Pair_emb_wo_templ(nn.Module):
    # TODO: embedding without template info
    def __init__(
        self, 
        d_model=128, 
        d_seq=21, 
        p_drop=0.1,
        network_2track=False,
    ):
        super(Pair_emb_wo_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.projection = nn.Linear(d_model + 1, d_model)
        self.netwok_2track = network_2track
        if not self.network_2track:
            self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)

    def forward(self, seq, idx):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        seq = self.emb(seq)  # (B, L, d_model//2)
        left = seq.unsqueeze(2).repeat(1, 1, L, 1)
        right = seq.unsqueeze(1).repeat(1, L, 1, 1)
        seqsep = torch.abs(idx[:, :, None] - idx[:, None, :]) + 1
        seqsep = torch.log(seqsep.float()).view(B, L, L, 1)
        
        pair = torch.cat((left, right, seqsep), dim=-1)
        pair = self.projection(pair)
        if not self.network_2track:
            pair = self.pos(pair, idx)
        return pair
