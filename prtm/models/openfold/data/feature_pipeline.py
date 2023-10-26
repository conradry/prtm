# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from prtm.models.openfold.config import DataConfig, Features
from prtm.models.openfold.data import input_pipeline

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    tensor_dict = {
        k: torch.tensor(v)
        for k, v in np_example.items()
        if k in features and v is not None
    }

    return tensor_dict


def np_example_to_features(
    np_example: Features,
    config: DataConfig,
):
    np_example = asdict(np_example)
    feature_names = config.common.unsupervised_features

    if config.common.use_templates:
        feature_names += config.common.template_features

    if "deletion_matrix_int" in np_example:
        np_example["deletion_matrix"] = np_example.pop("deletion_matrix_int").astype(
            np.float32
        )

    tensor_dict = np_to_tensor_dict(np_example=np_example, features=feature_names)
    with torch.no_grad():
        features = input_pipeline.process_tensors_from_config(
            tensor_dict,
            config.common,
            config.predict,
        )

    features["use_clamped_fape"] = torch.full(
        size=[config.common.max_recycling_iters + 1],
        fill_value=0.0,
        dtype=torch.float32,
    )

    return {k: v for k, v in features.items()}


class FeaturePipeline:
    def __init__(
        self,
        config: DataConfig,
    ):
        self.config = config

    def process_features(
        self,
        raw_features: Features,
    ) -> FeatureDict:
        return np_example_to_features(
            np_example=raw_features,
            config=self.config,
        )
