import contextlib
from functools import partial
from typing import List

import numpy as np
import torch


def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        try:
            return fn(tree)
        except:
            raise ValueError(f"cannot apply {fn} on {tree}.")
    else:
        raise ValueError(f"{type(tree)} not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def one_hot(x, num_classes, dtype=torch.float32):
    x_one_hot = torch.zeros(*x.shape, num_classes, dtype=dtype, device=x.device)
    x_one_hot.scatter_(-1, x.long().unsqueeze(-1), 1)
    return x_one_hot


def batched_gather(data, inds, dim=0, num_batch_dims=0):
    assert dim < 0 or dim - num_batch_dims >= 0
    ranges = []
    for i, s in enumerate(data.shape[:num_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - num_batch_dims)]
    remaining_dims[dim - num_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def checkpoint_sequential(
    functions,
    input,
    enabled=True,
):
    def wrap_tuple(a):
        return (a,) if type(a) is not tuple else a

    def exec(func, a):
        return wrap_tuple(func(*a))

    def get_wrap_exec(func):
        def wrap_exec(*a):
            return exec(func, a)

        return wrap_exec

    input = wrap_tuple(input)

    is_grad_enabled = torch.is_grad_enabled()

    if enabled and is_grad_enabled:
        for func in functions:
            input = torch.utils.checkpoint.checkpoint(get_wrap_exec(func), *input)
    else:
        for func in functions:
            input = exec(func, input)
    return input


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def str_hash(text: str):
    hash = 0
    for ch in text:
        hash = (hash * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
    return hash


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds, key=None):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return

    def check_seed(s):
        assert type(s) == int or type(s) == np.int32 or type(s) == np.int64

    check_seed(seed)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 1e8)
    if key is not None:
        seed = int(hash((seed, str_hash(key))) % 1e8)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collate_dict(
    values,
    dim=0,
):
    if len(values) <= 0:
        return values
    ret = {}
    keys = values[0].keys()
    for key in keys:
        ret[key] = torch.stack([v[key] for v in values], dim=dim)
    return ret
