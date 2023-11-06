###
#   Inspired by graph transformer implementation from https://github.com/lucidrains/graph-transformer-pytorch
###

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from prtm.models.igfold.utils.coordinates import get_ideal_coords, place_o_coords
from prtm.models.igfold.utils.general import default, exists
from prtm.models.igfold.utils.transforms import (
    quaternion_multiply,
    quaternion_to_matrix,
)
from torch import einsum, nn
from torch.cuda.amp import autocast

List = nn.ModuleList


class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn,
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        *args,
        **kwargs,
    ):
        x = self.norm(x)
        return self.fn(
            x,
            *args,
            **kwargs,
        )


# gated residual


class Residual(nn.Module):
    def forward(
        self,
        x,
        res,
    ):
        return x + res


class GatedResidual(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        edge_dim=None,
    ):
        super().__init__()
        edge_dim = default(
            edge_dim,
            dim,
        )

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(
            dim,
            inner_dim,
        )
        self.to_kv = nn.Linear(
            dim,
            inner_dim * 2,
        )
        self.edges_to_kv = nn.Linear(
            edge_dim,
            inner_dim,
        )

        self.to_out = nn.Linear(
            inner_dim,
            dim,
        )

    def forward(
        self,
        nodes,
        edges,
        mask=None,
    ):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(
            2,
            dim=-1,
        )

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(
            lambda t: rearrange(
                t,
                "b ... (h d) -> (b h) ... d",
                h=h,
            ),
            (q, k, v, e_kv),
        )

        ek, ev = e_kv, e_kv

        k, v = map(
            lambda t: rearrange(
                t,
                "b j d -> b () j d ",
            ),
            (k, v),
        )
        k = k + ek
        v = v + ev

        sim = (
            einsum(
                "b i d, b i j d -> b i j",
                q,
                k,
            )
            * self.scale
        )

        if exists(mask):
            mask = rearrange(
                mask,
                "b i -> b i ()",
            ) & rearrange(
                mask,
                "b j -> b () j",
            )
            mask = repeat(
                mask,
                "b ... -> (b h) ...",
                h=self.heads,
            )
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum(
            "b i j, b i j d -> b i d",
            attn,
            v,
        )
        out = rearrange(
            out,
            "(b h) n d -> b n (h d)",
            h=h,
        )
        return self.to_out(out)


def FeedForward(dim, ff_mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim),
    )


class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head=64,
        edge_dim=None,
        heads=8,
        with_feedforwards=False,
        norm_edges=False,
    ):
        super().__init__()
        self.layers = List([])
        edge_dim = default(
            edge_dim,
            dim,
        )
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        for _ in range(depth):
            self.layers.append(
                List(
                    [
                        List(
                            [
                                PreNorm(
                                    dim,
                                    Attention(
                                        dim,
                                        edge_dim=edge_dim,
                                        dim_head=dim_head,
                                        heads=heads,
                                    ),
                                ),
                                GatedResidual(dim),
                            ]
                        ),
                        List(
                            [
                                PreNorm(
                                    dim,
                                    FeedForward(dim),
                                ),
                                GatedResidual(dim),
                            ]
                        )
                        if with_feedforwards
                        else None,
                    ]
                )
            )

    def forward(
        self,
        nodes,
        edges,
        mask=None,
    ):
        edges = self.norm_edges(edges)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(
                attn(
                    nodes,
                    edges,
                    mask=mask,
                ),
                nodes,
            )

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(
                    ff(nodes),
                    nodes,
                )

        return nodes, edges


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value


# classes


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        scalar_key_dim=16,
        scalar_value_dim=16,
        point_key_dim=4,
        point_value_dim=4,
        pairwise_repr_dim=None,
        require_pairwise_repr=True,
        eps=1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr

        # num attention contributions

        num_attn_logits = 3 if require_pairwise_repr else 2

        # qkv projection for scalar attention (normal)

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias=False)

        # qkv projection for point attention (coordinate and orientation aware)

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.0)) - 1.0)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = (
            (num_attn_logits * point_key_dim) * (9 / 2)
        ) ** -0.5

        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias=False)

        # pairwise representation projection to attention bias

        pairwise_repr_dim = (
            default(pairwise_repr_dim, dim) if require_pairwise_repr else 0
        )

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits**-0.5

            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(pairwise_repr_dim, heads), Rearrange("b ... h -> (b h) ...")
            )

        # combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)

        self.to_out = nn.Linear(
            heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)),
            dim,
        )

    def forward(
        self, single_repr, pairwise_repr=None, *, rotations, translations, mask=None
    ):
        x, b, h, eps, require_pairwise_repr = (
            single_repr,
            single_repr.shape[0],
            self.heads,
            self.eps,
            self.require_pairwise_repr,
        )
        assert not (
            require_pairwise_repr and not exists(pairwise_repr)
        ), "pairwise representation must be given as second argument"

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways

        q_scalar, k_scalar, v_scalar = (
            self.to_scalar_q(x),
            self.to_scalar_k(x),
            self.to_scalar_v(x),
        )

        q_point, k_point, v_point = (
            self.to_point_q(x),
            self.to_point_k(x),
            self.to_point_v(x),
        )

        # split out heads

        q_scalar, k_scalar, v_scalar = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q_scalar, k_scalar, v_scalar),
        )
        q_point, k_point, v_point = map(
            lambda t: rearrange(t, "b n (h d c) -> (b h) n d c", h=h, c=3),
            (q_point, k_point, v_point),
        )

        rotations = repeat(rotations, "b n r1 r2 -> (b h) n r1 r2", h=h)
        translations = repeat(translations, "b n c -> (b h) n () c", h=h)

        # rotate qkv points into global frame

        q_point = (
            einsum("b n d c, b n c r -> b n d r", q_point, rotations) + translations
        )
        k_point = (
            einsum("b n d c, b n c r -> b n d r", k_point, rotations) + translations
        )
        v_point = (
            einsum("b n d c, b n c r -> b n d r", v_point, rotations) + translations
        )

        # derive attn logits for scalar and pairwise

        attn_logits_scalar = (
            einsum("b i d, b j d -> b i j", q_scalar, k_scalar)
            * self.scalar_attn_logits_scale
        )

        if require_pairwise_repr:
            attn_logits_pairwise = (
                self.to_pairwise_attn_bias(pairwise_repr)
                * self.pairwise_attn_logits_scale
            )

        # derive attn logits for point attention

        point_qk_diff = rearrange(q_point, "b i d c -> b i () d c") - rearrange(
            k_point, "b j d c -> b () j d c"
        )
        point_dist = (point_qk_diff**2).sum(dim=-2)

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, "h -> (b h) () () ()", b=b)

        attn_logits_points = -0.5 * (
            point_dist * point_weights * self.point_attn_logits_scale
        ).sum(dim=-1)

        # combine attn logits

        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise

        # mask

        if exists(mask):
            mask = rearrange(mask, "b i -> b i ()") * rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        # attention

        attn = attn_logits.softmax(dim=-1)

        with disable_tf32(), autocast(enabled=False):
            # disable TF32 for precision

            # aggregate values

            results_scalar = einsum("b i j, b j d -> b i d", attn, v_scalar)

            attn_with_heads = rearrange(attn, "(b h) i j -> b h i j", h=h)

            if require_pairwise_repr:
                results_pairwise = einsum(
                    "b h i j, b i j d -> b h i d", attn_with_heads, pairwise_repr
                )

            # aggregate point values

            results_points = einsum("b i j, b j d c -> b i d c", attn, v_point)

            # rotate aggregated point values back into local frame

            results_points = einsum(
                "b n d c, b n c r -> b n d r",
                results_points - translations,
                rotations.transpose(-1, -2),
            )
            results_points_norm = torch.sqrt(
                torch.square(results_points).sum(dim=-1) + eps
            )

        # merge back heads

        results_scalar = rearrange(results_scalar, "(b h) n d -> b n (h d)", h=h)
        results_points = rearrange(results_points, "(b h) n d c -> b n (h d c)", h=h)
        results_points_norm = rearrange(
            results_points_norm, "(b h) n d -> b n (h d)", h=h
        )

        results = (results_scalar, results_points, results_points_norm)

        if require_pairwise_repr:
            results_pairwise = rearrange(results_pairwise, "b h n d -> b n (h d)", h=h)
            results = (*results, results_pairwise)

        # concat results and project out

        results = torch.cat(results, dim=-1)
        return self.to_out(results)


# one transformer block based on IPA


def FeedForwardList(dim, mult=1.0, num_layers=2, act=nn.ReLU):
    layers = []
    dim_hidden = dim * mult

    for ind in range(num_layers):
        is_first = ind == 0
        is_last = ind == (num_layers - 1)
        dim_in = dim if is_first else dim_hidden
        dim_out = dim if is_last else dim_hidden

        layers.append(nn.Linear(int(dim_in), int(dim_out)))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)


class IPABlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult=1,
        ff_num_layers=3,  # in the paper, they used 3 layer transition (feedforward) block
        post_norm=True,  # in the paper, they used post-layernorm - offering pre-norm as well
        **kwargs,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = InvariantPointAttention(dim=dim, **kwargs)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForwardList(dim, mult=ff_mult, num_layers=ff_num_layers)

    def forward(self, x, **kwargs):
        post_norm = self.post_norm

        attn_input = x if post_norm else self.attn_norm(x)
        x = self.attn(attn_input, **kwargs) + x
        x = self.attn_norm(x) if post_norm else x

        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.ff_norm(x) if post_norm else x
        return x


class IPAEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        **kwargs,
    ):
        super().__init__()

        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                IPABlock(
                    dim=dim,
                    **kwargs,
                )
            )

    def forward(
        self,
        x,
        *,
        translations=None,
        rotations=None,
        pairwise_repr=None,
        mask=None,
    ):
        for block in self.layers:
            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations,
                mask=mask,
            )

        return x


class IPATransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        stop_rotation_grad=False,
        **kwargs,
    ):
        super().__init__()

        self.stop_rotation_grad = stop_rotation_grad

        self.quaternion_to_matrix = quaternion_to_matrix
        self.quaternion_multiply = quaternion_multiply

        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ipa_block = IPABlock(
                dim=dim,
                **kwargs,
            )
            linear = nn.Linear(dim, 6)
            torch.nn.init.zeros_(linear.weight.data)
            torch.nn.init.zeros_(linear.bias.data)
            self.layers.append(nn.ModuleList([ipa_block, linear]))

    def forward(
        self,
        single_repr,
        *,
        translations=None,
        quaternions=None,
        pairwise_repr=None,
        mask=None,
    ):
        x, device, quaternion_multiply, quaternion_to_matrix = (
            single_repr,
            single_repr.device,
            self.quaternion_multiply,
            self.quaternion_to_matrix,
        )
        b, n, *_ = x.shape

        # if no initial quaternions passed in, start from identity

        if not exists(quaternions):
            quaternions = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=device,
            )  # initial rotations
            quaternions = repeat(
                quaternions,
                "d -> b n d",
                b=b,
                n=n,
            )

        # if not translations passed in, start from identity

        if not exists(translations):
            translations = torch.zeros(
                (b, n, 3),
                device=device,
            )

        # go through the layers and apply invariant point attention and feedforward

        for block, to_update in self.layers:
            rotations = quaternion_to_matrix(quaternions)
            if self.stop_rotation_grad:
                rotations = rotations.detach()

            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations,
                mask=mask,
            )

            # update quaternion and translation

            quaternion_update, translation_update = to_update(x).chunk(
                2,
                dim=-1,
            )
            quaternion_update = F.pad(
                quaternion_update,
                (1, 0),
                value=1.0,
            )

            quaternions = quaternion_multiply(
                quaternions,
                quaternion_update,
            )
            translations = translations + einsum(
                "b n c, b n c r -> b n r",
                translation_update,
                rotations,
            )

        ideal_coords = get_ideal_coords().to(device)
        ideal_coords = repeat(
            ideal_coords,
            "a d -> b l a d",
            b=b,
            l=n,
        )

        rotations = quaternion_to_matrix(quaternions)
        points_global = einsum(
            "b n a c, b n c d -> b n a d",
            ideal_coords,
            rotations,
        ) + rearrange(
            translations,
            "b l d -> b l () d",
        )

        points_global = place_o_coords(points_global)

        return points_global, translations, quaternions


class TriangleGraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim,
        depth,
        gt_depth=1,
        gt_dim_head=32,
        gt_heads=8,
        tri_dim_hidden=None,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            graph_transformer = GraphTransformer(
                dim=dim,
                edge_dim=edge_dim,
                depth=gt_depth,
                heads=gt_heads,
                dim_head=gt_dim_head,
                with_feedforwards=True,
            )
            triangle_out = TriangleMultiplicativeModule(
                dim=edge_dim,
                hidden_dim=tri_dim_hidden,
                mix="outgoing",
            )
            triangle_in = TriangleMultiplicativeModule(
                dim=edge_dim,
                hidden_dim=tri_dim_hidden,
                mix="ingoing",
            )

            self.layers.append(
                nn.ModuleList([graph_transformer, triangle_out, triangle_in])
            )

    def forward(self, nodes, edges, mask=None):
        for gt, tri_out, tri_in in self.layers:
            if exists(mask):
                tri_mask = mask.unsqueeze(-2) & mask.unsqueeze(-1)
            else:
                tri_mask = None

            nodes, _ = gt(nodes, edges, mask=mask)
            edges = edges + tri_out(
                edges,
                mask=tri_mask,
            )
            edges = edges + tri_in(
                edges,
                mask=tri_mask,
            )

        return nodes, edges


class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim=None,
        mix="ingoing",
    ):
        super().__init__()
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"

        hidden_dim = default(
            hidden_dim,
            dim,
        )
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(
            dim,
            hidden_dim,
        )
        self.right_proj = nn.Linear(
            dim,
            hidden_dim,
        )

        self.left_gate = nn.Linear(
            dim,
            hidden_dim,
        )
        self.right_gate = nn.Linear(
            dim,
            hidden_dim,
        )
        self.out_gate = nn.Linear(dim, dim)

        # initialize all gating to be identity

        for gate in (
            self.left_gate,
            self.right_gate,
            self.out_gate,
        ):
            nn.init.constant_(
                gate.weight,
                0.0,
            )
            nn.init.constant_(
                gate.bias,
                1.0,
            )

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
        elif mix == "ingoing":
            self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(
            hidden_dim,
            dim,
        )

    def forward(self, x, mask=None):
        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"
        if exists(mask):
            mask = rearrange(
                mask,
                "b i j -> b i j ()",
            )

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(
            self.mix_einsum_eq,
            left,
            right,
        )

        out = self.to_out_norm(out)
        out = self.to_out(out)
        out = out * out_gate
        return out
