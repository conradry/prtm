# Copyright 2021 AlQuraishi Laboratory
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

import math
import unittest

import numpy as np
import torch
from proteome.models.folding.openfold.utils.rigid_utils import (
    Rigid,
    Rotation,
    quat_to_rot,
    rot_to_quat,
)

from .compare_utils import (
    alphafold_is_installed,
    import_alphafold,
    skip_unless_alphafold_installed,
)
from .config import consts

if alphafold_is_installed():
    alphafold = import_alphafold()
    import haiku as hk
    import jax


X_90_ROT = torch.tensor(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
)

X_NEG_90_ROT = torch.tensor(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]
)


def test_rigid_from_3_points_shape():
    batch_size = 2
    n_res = 5

    x1 = torch.rand((batch_size, n_res, 3))
    x2 = torch.rand((batch_size, n_res, 3))
    x3 = torch.rand((batch_size, n_res, 3))

    r = Rigid.from_3_points(x1, x2, x3)

    rot, tra = r.get_rots().get_rot_mats(), r.get_trans()

    assert rot.shape == (batch_size, n_res, 3, 3)
    assert torch.all(tra == x2)


def test_rigid_from_4x4():
    batch_size = 2
    transf = [
        [1, 0, 0, 1],
        [0, 0, -1, 2],
        [0, 1, 0, 3],
        [0, 0, 0, 1],
    ]
    transf = torch.tensor(transf)

    true_rot = transf[:3, :3]
    true_trans = transf[:3, 3]

    transf = torch.stack([transf for _ in range(batch_size)], dim=0)

    r = Rigid.from_tensor_4x4(transf)

    rot, tra = r.get_rots().get_rot_mats(), r.get_trans()

    assert torch.all(rot == true_rot.unsqueeze(0))
    assert torch.all(tra == true_trans.unsqueeze(0))


def test_rigid_shape():
    batch_size = 2
    n = 5
    transf = Rigid(
        Rotation(rot_mats=torch.rand((batch_size, n, 3, 3))),
        torch.rand((batch_size, n, 3)),
    )

    assert transf.shape == (batch_size, n)


def test_rigid_cat():
    batch_size = 2
    n = 5
    transf = Rigid(
        Rotation(rot_mats=torch.rand((batch_size, n, 3, 3))),
        torch.rand((batch_size, n, 3)),
    )

    transf_cat = Rigid.cat([transf, transf], dim=0)

    transf_rots = transf.get_rots().get_rot_mats()
    transf_cat_rots = transf_cat.get_rots().get_rot_mats()

    assert transf_cat_rots.shape == (batch_size * 2, n, 3, 3)

    transf_cat = Rigid.cat([transf, transf], dim=1)
    transf_cat_rots = transf_cat.get_rots().get_rot_mats()

    assert transf_cat_rots.shape == (batch_size, n * 2, 3, 3)

    assert torch.all(transf_cat_rots[:, :n] == transf_rots)
    assert torch.all(transf_cat.get_trans()[:, :n] == transf.get_trans())


def test_rigid_compose():
    trans_1 = [0, 1, 0]
    trans_2 = [0, 0, 1]

    r = Rotation(rot_mats=X_90_ROT)
    t = torch.tensor(trans_1)

    t1 = Rigid(Rotation(rot_mats=X_90_ROT), torch.tensor(trans_1))
    t2 = Rigid(Rotation(rot_mats=X_NEG_90_ROT), torch.tensor(trans_2))

    t3 = t1.compose(t2)

    assert torch.all(t3.get_rots().get_rot_mats() == torch.eye(3))
    assert torch.all(t3.get_trans() == 0)


def test_rigid_apply():
    rots = torch.stack([X_90_ROT, X_NEG_90_ROT], dim=0)
    trans = torch.tensor([1, 1, 1])
    trans = torch.stack([trans, trans], dim=0)

    t = Rigid(Rotation(rot_mats=rots), trans)

    x = torch.arange(30)
    x = torch.stack([x, x], dim=0)
    x = x.view(2, -1, 3)  # [2, 10, 3]

    pts = t[..., None].apply(x)

    # All simple consequences of the two x-axis rotations
    assert torch.all(pts[..., 0] == x[..., 0] + 1)
    assert torch.all(pts[0, :, 1] == x[0, :, 2] * -1 + 1)
    assert torch.all(pts[1, :, 1] == x[1, :, 2] + 1)
    assert torch.all(pts[0, :, 2] == x[0, :, 1] + 1)
    assert torch.all(pts[1, :, 2] == x[1, :, 1] * -1 + 1)


def test_quat_to_rot():
    forty_five = math.pi / 4
    quat = torch.tensor([math.cos(forty_five), math.sin(forty_five), 0, 0])
    rot = quat_to_rot(quat)
    eps = 1e-07
    assert torch.all(torch.abs(rot - X_90_ROT) < eps)


def test_rot_to_quat():
    quat = rot_to_quat(X_90_ROT)
    eps = 1e-07
    ans = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
    assert torch.all(torch.abs(quat - ans) < eps)


@skip_unless_alphafold_installed()
def test_pre_compose_compare():
    quat = np.random.rand(20, 4)
    trans = [np.random.rand(20) for _ in range(3)]
    quat_affine = alphafold.model.quat_affine.QuatAffine(quat, translation=trans)

    update_vec = np.random.rand(20, 6)
    new_gt = quat_affine.pre_compose(update_vec)

    quat_t = torch.tensor(quat)
    trans_t = torch.stack([torch.tensor(t) for t in trans], dim=-1)
    rigid = Rigid(Rotation(quats=quat_t), trans_t)
    new_repro = rigid.compose_q_update_vec(torch.tensor(update_vec))

    new_gt_q = torch.tensor(np.array(new_gt.quaternion))
    new_gt_t = torch.stack(
        [torch.tensor(np.array(t)) for t in new_gt.translation], dim=-1
    )
    new_repro_q = new_repro.get_rots().get_quats()
    new_repro_t = new_repro.get_trans()

    assert torch.max(torch.abs(new_gt_q - new_repro_q)) < consts.eps
    assert torch.max(torch.abs(new_gt_t - new_repro_t)) < consts.eps
