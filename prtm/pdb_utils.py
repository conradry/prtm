import io

import numpy as np
from Bio import PDB
from prtm.constants import residue_constants

try:
    # openmm >= 7.6
    from openmm import app as openmm_app
    from openmm.app.internal.pdbstructure import PdbStructure
except ImportError:
    # openmm < 7.6 (requires DeepMind patch)
    from simtk.openmm import app as openmm_app
    from simtk.openmm.app.internal.pdbstructure import PdbStructure


def overwrite_pdb_coordinates(pdb_str: str, pos) -> str:
    """Overwrites the coordinates in pdb_str with contents of pos array."""
    pdb_file = io.StringIO(pdb_str)
    structure = PdbStructure(pdb_file)
    topology = openmm_app.PDBFile(structure).getTopology()
    with io.StringIO() as f:
        openmm_app.PDBFile.writeFile(topology, pos, f)
        return f.getvalue()


def overwrite_b_factors(pdb_str: str, bfactors: np.ndarray) -> str:
    """Overwrites the B-factors in pdb_str with contents of bfactors array.

    Args:
      pdb_str: An input PDB string.
      bfactors: A numpy array with shape [1, n_residues, 37]. We assume that the
        B-factors are per residue; i.e. that the nonzero entries are identical in
        [0, i, :].

    Returns:
      A new PDB string with the B-factors replaced.
    """
    if bfactors.shape[-1] != residue_constants.atom_type_num:
        raise ValueError(
            f"Invalid final dimension size for bfactors: {bfactors.shape[-1]}."
        )

    parser = PDB.PDBParser(QUIET=True)
    handle = io.StringIO(pdb_str)
    structure = parser.get_structure("", handle)

    curr_resid = ("", "", "")
    idx = -1
    for atom in structure.get_atoms():
        atom_resid = atom.parent.get_id()
        if atom_resid != curr_resid:
            idx += 1
            if idx >= bfactors.shape[0]:
                raise ValueError(
                    "Index into bfactors exceeds number of residues. "
                    "B-factors shape: {shape}, idx: {idx}."
                )
        curr_resid = atom_resid
        atom.bfactor = bfactors[idx, residue_constants.atom_order["CA"]]

    new_pdb = io.StringIO()
    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(new_pdb)

    return new_pdb.getvalue()