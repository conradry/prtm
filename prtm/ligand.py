from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformer(mol: Chem.Mol):
    """Generates a conformer for the molecule using ETKDGv2."""
    mol = deepcopy(mol)
    ps = AllChem.ETKDGv2()
    failures, id = 0, -1
    while failures < 3 and id == -1:
        if failures > 0:
            print(f"rdkit coords could not be generated. trying again {failures}.")
        id = AllChem.EmbedMolecule(mol, ps)
        if id != -1:
            return mol
        else:
            failures += 1

    raise Exception("Failed to generate conformer!")


def load_molecule_from_file(
    molecule_file: str,
    sanitize: bool = False,
    remove_hs: bool = False,
) -> Chem.Mol:
    """Load a molecule from a file.

    Args:
        molecule_file: The path to the molecule file.
        sanitize: Whether to sanitize the molecule.
        remove_hs: Whether to remove hydrogens.

    Returns:
        The RDKit molecule.
    """
    if molecule_file.endswith(".mol2"):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith(".pdbqt"):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()

        pdb_block = ""
        for line in pdbqt_data:
            pdb_block += "{}\n".format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError(
            "Expect the format of the molecule_file to be "
            f"one of .mol2, .sdf, .pdbqt and .pdb, got {molecule_file}"
        )

    if sanitize:
        Chem.SanitizeMol(mol)
    if remove_hs:
        mol = Chem.RemoveHs(mol, sanitize=sanitize)

    return mol
