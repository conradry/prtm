import copy as copy_lib
import functools
import gzip
import json
import pickle
from typing import *

import numpy as np
from scipy import sparse as sp

from prtm.models.unifold.data import residue_constants as rc
from prtm.models.unifold.data.data_ops import NumpyDict


def lru_cache(maxsize=16, typed=False, copy=False, deepcopy=False):
    if deepcopy:

        def decorator(f):
            cached_func = functools.lru_cache(maxsize, typed)(f)

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return copy_lib.deepcopy(cached_func(*args, **kwargs))

            return wrapper

    elif copy:

        def decorator(f):
            cached_func = functools.lru_cache(maxsize, typed)(f)

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return copy_lib.copy(cached_func(*args, **kwargs))

            return wrapper

    else:
        decorator = functools.lru_cache(maxsize, typed)
    return decorator


def uncompress_features(feats: NumpyDict) -> NumpyDict:
    if "sparse_deletion_matrix_int" in feats:
        v = feats.pop("sparse_deletion_matrix_int")
        v = to_dense_matrix(v)
        feats["deletion_matrix"] = v
    return feats


@lru_cache(maxsize=8, deepcopy=True)
def load_pickle_safe(path: str) -> Dict[str, Any]:
    def load(path):
        assert path.endswith(".pkl") or path.endswith(
            ".pkl.gz"
        ), f"bad suffix in {path} as pickle file."
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rb") as f:
            return pickle.load(f)

    ret = load(path)
    ret = uncompress_features(ret)
    return ret


@lru_cache(maxsize=8, copy=True)
def load_pickle(path: str) -> Dict[str, Any]:
    def load(path):
        assert path.endswith(".pkl") or path.endswith(
            ".pkl.gz"
        ), f"bad suffix in {path} as pickle file."
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rb") as f:
            return pickle.load(f)

    ret = load(path)
    ret = uncompress_features(ret)
    return ret


def correct_template_restypes(feature):
    """Correct template restype to have the same order as residue_constants."""
    feature = np.argmax(feature, axis=-1).astype(np.int32)
    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
    return feature


def convert_all_seq_feature(feature: NumpyDict) -> NumpyDict:
    feature["msa"] = feature["msa"].astype(np.uint8)
    if "num_alignments" in feature:
        feature.pop("num_alignments")
    make_all_seq_key = lambda k: f"{k}_all_seq" if not k.endswith("_all_seq") else k
    return {make_all_seq_key(k): v for k, v in feature.items()}


def to_dense_matrix(spmat_dict: NumpyDict):
    spmat = sp.coo_matrix(
        (spmat_dict["data"], (spmat_dict["row"], spmat_dict["col"])),
        shape=spmat_dict["shape"],
        dtype=np.float32,
    )
    return spmat.toarray()


FEATS_DTYPE = {"msa": np.int32}


def filter(feature: NumpyDict, **kwargs) -> NumpyDict:
    assert len(kwargs) == 1, f"wrong usage of filter with kwargs: {kwargs}"
    if "desired_keys" in kwargs:
        feature = {k: v for k, v in feature.items() if k in kwargs["desired_keys"]}
    elif "required_keys" in kwargs:
        for k in kwargs["required_keys"]:
            assert k in feature, f"cannot find required key {k}."
    elif "ignored_keys" in kwargs:
        feature = {k: v for k, v in feature.items() if k not in kwargs["ignored_keys"]}
    else:
        raise AssertionError(f"wrong usage of filter with kwargs: {kwargs}")
    return feature
