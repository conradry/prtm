# Copyright Generate Biomedicines, Inc.
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

import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from prtm.models import chroma
from prtm.models.chroma.structure import backbone
from sklearn.decomposition import PCA


def letter_to_point_cloud(
    letter="G",
    width_pixels=35,
    font=os.path.join(
        os.path.dirname(chroma.__path__[0]), "assets/LiberationSans-Regular.ttf"
    ),
    depth_ratio=0.15,
    fontsize_ratio=1.2,
    stroke_width=1,
    margin=0.5,
    max_points=2000,
):
    """Build a point cloud from a letter"""
    depth = int(depth_ratio * width_pixels)
    fontsize = int(fontsize_ratio * width_pixels)

    font = ImageFont.truetype(font, fontsize)
    ascent, descent = font.getmetrics()
    text_width = font.getmask(letter).getbbox()[2]
    text_height = font.getmask(letter).getbbox()[3] + descent

    margin_width = int(text_width * margin)
    margin_height = int(text_height * margin)
    image_size = [text_width + margin_width, text_height + margin_height]

    image = Image.new("RGBA", image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text(
        (margin_width // 2, margin_height // 2),
        letter,
        (0, 0, 0),
        font=font,
        stroke_width=stroke_width,
        stroke_fill="black",
    )

    A = np.asarray(image).mean(-1)
    A = A < 100.0
    V = np.ones(list(A.shape[:2]) + [depth]) * A[:, :, None]
    X_point_cloud = np.stack(np.nonzero(V), 1)
    # Uniform dequantization
    X_point_cloud = X_point_cloud + np.random.rand(*X_point_cloud.shape)

    if max_points is not None and X_point_cloud.shape[0] > max_points:
        np.random.shuffle(X_point_cloud)
        X_point_cloud = X_point_cloud[:max_points, :]

    return X_point_cloud


def point_cloud_rescale(
    X, num_residues, neighbor_k=8, volume_per_residue=128.57, scale_ratio=0.4
):
    """Rescale target coordinates to occupy protein-sized volume"""

    # Use heuristic for radius value from the average m-th nearest neighbor
    # This was tuned empirically for target problems (could be optimized on the fly as LB estimate)
    D = np.sqrt(np.square(X[None, :] - X[:, None]).sum(-1))
    radius = 0.5 * np.sort(D, axis=1)[:, neighbor_k].mean()
    D.max()

    # Estimate initial volume with 2nd order inclusion exclusion
    V = point_cloud_volume(X, radius)

    # Compute target volume, which scales linearly with number of residues
    V_target = num_residues * volume_per_residue
    scale_factor = (scale_ratio * V_target / V) ** (1.0 / 3.0)
    X_rescale = scale_factor * X
    cutoff_D = scale_factor * radius
    return X_rescale, cutoff_D


def point_cloud_volume(X, radius):
    """Estimate the volume of a point cloud given sphere radii"""
    N = X.shape[0]

    # Volume estimation - One body volumes
    V_1 = N * (4.0 / 3.0) * np.pi * radius**3

    # Volume estimation - 2nd order overlaps
    D = np.sqrt(np.square(X[None, :] - X[:, None]).sum(-1))
    overlap_ij = (
        (D < 2.0 * radius)
        * (np.pi / 12.0)
        * (4.0 * radius + D)
        * (2.0 * radius - D) ** 2
    )
    V_2 = np.tril(overlap_ij, k=-1).sum()

    # Inclusion-Exclusion Principle
    volume = V_1 - V_2
    return volume


def plane_split_protein(X=None, C=None, protein=None, mask_percent=0.5):
    if protein is None:
        assert X is not None and C is not None
    else:
        X, C, _ = protein.to_XCS()
        X = X[:, :, :4, :]

    X = backbone.center_X(X, C)
    points = X[C > 0].reshape(-1, 3)
    pca = PCA(n_components=1)
    normal = torch.from_numpy(
        pca.fit_transform(points.detach().cpu().numpy().transpose(1, 0))
    ).to(X.device)
    c_alphas = X[:, :, 1, :]

    c = 0
    tries = 0

    def percent_masked(c):
        C_mask = ((c_alphas @ normal) > c).squeeze(-1) & (C > 0)
        return (~C_mask).float().sum().item() / (C > 0).sum().item()

    # In the first stage we find the minimum C such that all of the residues
    # lie on one side of the plane (c_alphas @ normal = c)
    while (percent_masked(c) < 1.0) and (tries < 300000):
        tries += 1
        c += 100

    # Now we drag the plane back until percent_masked - masked_percent is small.
    size = X.size(1)
    threshold = 0.1 if size < 100 else 0.05 if size < 500 else 0.01
    tries = 0
    while (np.abs(percent_masked(c) - mask_percent) > threshold) and (tries < 300000):
        c -= 100
        tries += 1

    if tries >= 300000:
        print(
            "Tried and failed to split protein by plane to grab"
            f" {mask_percent} residues."
        )
        c = 0
        C_mask = ((c_alphas @ normal) > c).squeeze(-1) & (C > 0)
        print(
            f"Returning {100 * percent_masked(0.0):.2f} percent residues masked"
            " instead."
        )

    else:
        C_mask = ((c_alphas @ normal) > c).squeeze(-1) & (C > 0)
        print(
            f"Split protein by plane, masking {100 * percent_masked(c):.2f} percent of"
            " residues."
        )

    return C_mask
