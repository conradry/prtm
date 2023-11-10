from typing import Optional

import numpy as np
import torch
from prtm import protein
from prtm.constants.residue_constants import proteinmppn_restypes


def featurize_structure(
    structure: protein.Protein4,
    device: Optional[torch.device] = torch.device("cpu"),
):
    """Pack and pad batch into torch tensors"""
    assert isinstance(
        structure, protein.Protein4
    ), "Structure must be a Protein4 object"

    sequence = structure.sequence()
    length = len(sequence)

    X = structure.atom_positions
    score = 100 * np.ones([1, length])
    S = np.asarray([proteinmppn_restypes.index(a) for a in sequence], dtype=np.int32)

    # Add batch dimension
    X = X[None]
    S = S[None]

    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)  # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X) + np.nan
    for i, n in enumerate(numbers):
        X_new[i, :n, ::] = X[i][mask[i] == 1]
        S_new[i, :n] = S[i][mask[i] == 1]

    X = X_new
    S = S_new

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long).to(device)
    score = torch.from_numpy(score).float().to(device)
    X = torch.from_numpy(X).to(dtype=torch.float32).to(device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32).to(device)

    return X, S, score, mask
