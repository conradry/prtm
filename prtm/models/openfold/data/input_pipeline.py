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

from functools import partial
from typing import Dict

import torch
from prtm.models.openfold.config import SHAPE_SCHEMA, CommonData, PredictData
from prtm.models.openfold.data import data_transforms


def nonensembled_transform_fns(common_cfg: CommonData):
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.correct_msa_restypes,
        data_transforms.squeeze_features,
        data_transforms.randomly_replace_msa_with_unknown(0.0),
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
        data_transforms.make_hhblits_profile,
    ]
    if common_cfg.use_templates:
        transforms.extend(
            [
                data_transforms.fix_templates_aatype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta("template_"),
            ]
        )
        if common_cfg.use_template_torsion_angles:
            transforms.extend(
                [
                    data_transforms.atom37_to_torsion_angles("template_"),
                ]
            )

    transforms.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )

    return transforms


def ensembled_transform_fns(
    common_cfg: CommonData,
    mode_cfg: PredictData,
    ensemble_seed: int,
):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = mode_cfg.max_extra_msa

    msa_seed = None
    if not common_cfg.resample_msa_in_recycling:
        msa_seed = ensemble_seed

    transforms.append(
        data_transforms.sample_msa(
            max_msa_clusters,
            keep_extra=True,
            seed=msa_seed,
        )
    )

    transforms.append(
        data_transforms.make_masked_msa(
            common_cfg.masked_msa, mode_cfg.masked_msa_replace_fraction
        )
    )

    if common_cfg.msa_cluster_features:
        transforms.append(data_transforms.nearest_neighbor_clusters())
        transforms.append(data_transforms.summarize_clusters())

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(data_transforms.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(data_transforms.delete_extra_msa)

    transforms.append(data_transforms.make_msa_feat())

    transforms.append(data_transforms.select_feat(list(SHAPE_SCHEMA.keys())))
    transforms.append(
        data_transforms.random_crop_to_size(
            mode_cfg.crop_size,
            mode_cfg.max_templates,
            SHAPE_SCHEMA,
            mode_cfg.subsample_templates,
            seed=ensemble_seed + 1,
        )
    )
    transforms.append(
        data_transforms.make_fixed_size(
            SHAPE_SCHEMA,
            pad_msa_clusters,
            mode_cfg.max_extra_msa,
            mode_cfg.crop_size,
            mode_cfg.max_templates,
        )
    )
    transforms.append(data_transforms.crop_templates(mode_cfg.max_templates))

    return transforms


def process_tensors_from_config(
    tensors: Dict[str, torch.Tensor],
    common_cfg: CommonData,
    mode_cfg: PredictData,
):
    """Based on the config, apply filters and transformations to the data."""

    ensemble_seed = torch.Generator().seed()

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(
            common_cfg,
            mode_cfg,
            ensemble_seed,
        )
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    nonensembled = nonensembled_transform_fns(common_cfg)

    tensors = compose(nonensembled)(tensors)

    if "no_recycling_iters" in tensors:
        num_recycling = int(tensors["no_recycling_iters"])
    else:
        num_recycling = common_cfg.max_recycling_iters

    tensors = map_fn(
        lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
    )

    return tensors


@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict