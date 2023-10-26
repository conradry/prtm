# pylint: disable=C,E1101,E1102
"""
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
"""
import numpy as np
import torch

try:
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
except:
    pass


class torch_default_dtype:
    def __init__(self, dtype):
        self.saved_dtype = None
        self.dtype = dtype

    def __enter__(self):
        self.saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self.saved_dtype)


def rot_z(gamma):
    """
    Rotation around Z axis
    """
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=torch.get_default_dtype())
    return torch.tensor(
        [
            [torch.cos(gamma), -torch.sin(gamma), 0],
            [torch.sin(gamma), torch.cos(gamma), 0],
            [0, 0, 1],
        ],
        dtype=gamma.dtype,
    )


def rot_y(beta):
    """
    Rotation around Y axis
    """
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.get_default_dtype())
    return torch.tensor(
        [
            [torch.cos(beta), 0, torch.sin(beta)],
            [0, 1, 0],
            [-torch.sin(beta), 0, torch.cos(beta)],
        ],
        dtype=beta.dtype,
    )


def rot(alpha, beta, gamma):
    """
    ZYZ Eurler angles rotation
    """
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def x_to_alpha_beta(x):
    """
    Convert point (x, y, z) on the sphere into (alpha, beta)
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.get_default_dtype())
    x = x / torch.norm(x)
    beta = torch.acos(x[2])
    alpha = torch.atan2(x[1], x[0])
    return (alpha, beta)


# These functions (x_to_alpha_beta and rot) satisfies that
# rot(*x_to_alpha_beta([x, y, z]), 0) @ np.array([[0], [0], [1]])
# is proportional to
# [x, y, z]


def irr_repr(order, alpha, beta, gamma, dtype=None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    return torch.tensor(
        wigner_D_matrix(order, np.array(alpha), np.array(beta), np.array(gamma)),
        dtype=torch.get_default_dtype() if dtype is None else dtype,
    )


def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.0])
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = torch.atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c


def kron(x, y):
    assert x.ndimension() == 2
    assert y.ndimension() == 2
    return torch.einsum("ij,kl->ikjl", (x, y)).view(
        x.size(0) * y.size(0), x.size(1) * y.size(1)
    )
