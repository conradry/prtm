from typing import Tuple, Optional

import numpy as np
import torch

from prtm.constants import residue_constants
from prtm.protein import ProteinBase


def dihedral(
    a: np.ndarray, 
    b: np.ndarray, 
    c: np.ndarray,
    d: np.ndarray,
    radians: bool = False,
):
    """Computes the dihedral angle formed by four 3D points represented by AtomLocationView objects.

    Args:
        a1, a2, a3, a4 (AtomLocationView): four 3D points.
        radian (bool, optional): if True (default False), will return the angle in radians.
            Otherwise, in degrees.

    Returns:
        Dihedral angle `a1`-`a2`-`a3`-`a4`.
    """
    AB = a - b
    CB = c - b
    DC = d - c

    if min([np.linalg.norm(p) for p in [AB, CB, DC]]) == 0.0:
        raise Exception("some points coincide in dihedral calculation")

    ABxCB = np.cross(AB, CB)
    ABxCB = ABxCB / np.linalg.norm(ABxCB)
    DCxCB = np.cross(DC, CB)
    DCxCB = DCxCB / np.linalg.norm(DCxCB)

    # the following is necessary for values very close to 1 but just above
    dotp = np.dot(ABxCB, DCxCB)
    if dotp > 1.0:
        dotp = 1.0
    elif dotp < -1.0:
        dotp = -1.0

    angle = np.arccos(dotp)
    if np.dot(ABxCB, DC) > 0:
        angle *= -1
    if not radians:
        angle *= 180.0 / np.pi

    return angle


def canonicalize_structure(prot: ProteinBase) -> ProteinBase:
    """
    Returns the canonical chroma structure for the given protein.
    """
    # Make sure the structure is in 37 atom format
    prot = prot.to_protein37()
    prot = prot.to_numpy()

    # Get the index for arginine
    arg_res_idx = residue_constants.restype_3["ARG"]
    arg_indices = np.where(prot.aatype == arg_res_idx)[0]

    # Check if sidechain atoms are present in the structure
    arg_sidechain_atom_indices = np.array(
        [residue_constants.atom_types.index(atom) for atom in ["CD", "NE", "CZ", "NH1", "NH2"]]
    )
    has_sidechain_atoms = np.where(
        prot.atom_mask[:, arg_sidechain_atom_indices].sum(axis=1) == 5
    )[0]

    arg_indices_with_sidechains = np.intersect1d(arg_indices, has_sidechain_atoms)
    for index in arg_indices_with_sidechains:
        dihe1 = dihedral(
            prot.atom_positions[index, arg_sidechain_atom_indices[0]],
            prot.atom_positions[index, arg_sidechain_atom_indices[1]],
            prot.atom_positions[index, arg_sidechain_atom_indices[2]],
            prot.atom_positions[index, arg_sidechain_atom_indices[3]],
        )
        dihe2 = dihedral(
            prot.atom_positions[index, arg_sidechain_atom_indices[0]],
            prot.atom_positions[index, arg_sidechain_atom_indices[1]],
            prot.atom_positions[index, arg_sidechain_atom_indices[2]],
            prot.atom_positions[index, arg_sidechain_atom_indices[4]],
        )
        if abs(dihe1) > abs(dihe2):
            # Swap NH1 and NH2 positions
            nh1_pos = prot.atom_positions[index, arg_sidechain_atom_indices[3]]
            nh2_pos = prot.atom_positions[index, arg_sidechain_atom_indices[4]]
            prot.atom_positions[index, arg_sidechain_atom_indices[3]] = nh2_pos
            prot.atom_positions[index, arg_sidechain_atom_indices[4]] = nh1_pos

    # TODO: Convert back to the original prot atom type?
    return prot


def protein_to_xcs(
    prot: ProteinBase,
    all_atom: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Convert System object to XCS format.

    `C` tensor has shape [num_residues], where it codes positions as 0
    when masked, positive integers for chain indices, and negative integers
    to represent missing residues of the corresponding positive integers.

    `S` tensor has shape [num_residues], it will map residue amino acid to alphabet integers.
    If it is not found in `alphabet`, it will default to `unknown_token`. Set `mask_unknown` to true if
    also want to mask `unk residue` in `chain_map`

    This function takes into account missing residues and updates chain_map
    accordingly.

    Args:
        system (type): generate System object to convert.
        all_atom (bool): Include side chain atoms. Default is `False`.
        batch_dimension (bool): Include a batch dimension. Default is `True`.
        mask_unknown (bool): Mask residues not found in the alphabet. Default is
            `True`.
        unknown_token (int): Default token index if a residue is not found in
            the alphabet. Default is `0`.
        reorder_chain (bool): If set to true will start indexing chain at 1,
            else will use the alphabet index (Default: True)
        altenate_alphabet (str): Alternative alphabet if not `None`.
        alternate_atoms (list): Alternate atom name subset for `X` if not `None`.
        get_indices (bool): Also return the location indices corresponding to the
            returned `X` tensor.

    Returns:
        X (torch.Tensor): Coordinates with shape `(1, num_residues, num_atoms, 3)`.
            `num_atoms` will be 14 if `all_atom=True` or 4 otherwise.
        C (torch.LongTensor): Chain map with shape `(1, num_residues)`. It codes
            positions as 0 when masked, positive integers for chain indices,
            and negative integers to represent missing residues of the
            corresponding positive integers.
        S (torch.LongTensor): Sequence with shape `(1, num_residues)`.
        location_indices (np.ndaray, optional): location indices corresponding to
            the coordinates in `X`.

    """
    # Either all_atom (backbone plus sidechains is 14 atoms)
    # or the backbone (4 atoms)
    if all_atom:
        prot = prot.to_protein14()
    else:
        prot = prot.to_protein4()

    prot = prot.to_numpy()

    # We want to reorder the chain_ids to start at 1 and be sequential
    chain_reordering = {
        chain_id: i for i, chain_id in enumerate(np.unique(prot.chain_index), 1)
    }
    C = np.vectorize(chain_reordering.get)(prot.chain_index)
    X = prot.atom_positions
    S = prot.aatype

    # Unknown residues are mapped to 0 for Chroma
    # but in prtm parsing they will be marked as X, 
    # so make the conversion
    masked_res_idx = residue_constants.restypes_with_x.index("X")
    S[S == masked_res_idx] = 0

    # Map from the default protein restypes to the alphabetical restypes
    chroma_alphabet = residue_constants.alphabetical_restypes
    restype_to_alphabetical = {
        res_idx: chroma_alphabet.index(res) 
        for res_idx,res in enumerate(residue_constants.restypes)
    }
    # Map the restypes to the alphabetical restypes
    S = np.vectorize(restype_to_alphabetical.get)(S)

    # Tensor everything and add batch dimension
    if device is None:
        device = torch.device("cpu")

    X = torch.tensor(X, device=device).float()[None]
    C = torch.tensor(C, device=device).type(torch.long)[None]
    S = torch.tensor(S, device=device).type(torch.long)[None]

    return X, C, S