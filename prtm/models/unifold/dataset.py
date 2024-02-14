import copy
import json
from typing import *

import ml_collections as mlc
from prtm.models.unifold.data.data_ops import NumpyDict, TorchDict

Rotation = Iterable[Iterable]
Translation = Iterable
Operation = Union[str, Tuple[Rotation, Translation]]
NumpyExample = Tuple[NumpyDict, Optional[List[NumpyDict]]]
TorchExample = Tuple[TorchDict, Optional[List[TorchDict]]]


def make_data_config(
    config: mlc.ConfigDict,
    mode: str,
    num_res: int,
    use_templates: bool = False,
    is_multimer: bool = False,
    is_supervised: bool = False,
) -> Tuple[mlc.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res
    feature_names = cfg.common.unsupervised_features + cfg.common.recycling_features
    if use_templates:
        feature_names += cfg.common.template_features
    if is_multimer:
        feature_names += cfg.common.multimer_features
    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names
