# By David T. Jones, June 2021

import torch


def reweight(msa1hot, cutoff):
    """Reweight MSA based on cutoff from
    https://github.com/gjoni/trRosetta/blob/master/network/utils.py
    """
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum("ikl,jkl->ij", msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1.0 / id_mask.float().sum(dim=-1)
    return w


def fast_dca(msa1hot, weights, penalty=4.5):
    """Shrunk covariance inversion from
    https://github.com/gjoni/trRosetta/blob/master/network/utils.py
    """
    device = msa1hot.device
    nr, nc, ns = msa1hot.shape
    x = msa1hot.view(nr, -1)
    num_points = weights.sum() - torch.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
    x = (x - mean) * torch.sqrt(weights[:, None])

    cov = (x.t() @ x) / num_points
    cov_reg = cov + torch.eye(nc * ns, device=device) * penalty / torch.sqrt(
        weights.sum()
    )

    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.transpose(1, 2).contiguous()
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum(dim=(1, 3))) * (
        1 - torch.eye(nc, device=device)
    )
    apc = x3.sum(dim=0, keepdims=True) * x3.sum(dim=1, keepdims=True) / x3.sum()  # type: ignore
    contacts = (x3 - apc) * (1 - torch.eye(nc, device=device))
    return torch.cat((features, contacts[:, :, None]), dim=2)
