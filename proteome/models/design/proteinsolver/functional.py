import torch
from torch.nn.functional import dropout, linear
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as sparse_softmax


def sparse_multi_head_attention_forward(
    query,
    key,
    value,
    index,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        index: ....
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source
          sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence
          length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(
                key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)]
            )
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k])
            v = torch.cat([v, bias_v])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [
                        attn_mask,
                        torch.zeros(
                            (attn_mask.size(0), 1),
                            dtype=attn_mask.dtype,
                            device=attn_mask.device,
                        ),
                    ],
                    dim=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(
                            (key_padding_mask.size(0), 1),
                            dtype=key_padding_mask.dtype,
                            device=key_padding_mask.device,
                        ),
                    ],
                    dim=1,
                )
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == 1
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                ),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                ),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = torch.cat(
                [
                    attn_mask,
                    torch.zeros(
                        (attn_mask.size(0), 1),
                        dtype=attn_mask.dtype,
                        device=attn_mask.device,
                    ),
                ],
                dim=1,
            )
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    torch.zeros(
                        (key_padding_mask.size(0), 1),
                        dtype=key_padding_mask.dtype,
                        device=key_padding_mask.device,
                    ),
                ],
                dim=1,
            )

    attn_output_weights = (q[:, index[0], :] * k).sum(dim=-1)
    assert list(attn_output_weights.size()) == [num_heads, src_len]

    # attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # From dense
    # assert list(attn_output_weights.size()) == [num_heads, tgt_len, src_len]  # From dense

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(1, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )
        attn_output_weights = attn_output_weights.view(num_heads, tgt_len, src_len)

    attn_output_weights = sparse_softmax(
        attn_output_weights.transpose(0, 1), index[0]
    ).transpose(0, 1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = scatter(
        attn_output_weights.unsqueeze(-1) * v, index[0], dim=1, reduce="sum"
    )
    # attn_output = torch.bmm(attn_output_weights, v)  # From dense
    assert list(attn_output.size()) == [num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        # attn_output_weights = attn_output_weights.view(num_heads, tgt_len, src_len)  # From dense
        return attn_output, attn_output_weights.sum(dim=0) / num_heads
    else:
        return attn_output, None
