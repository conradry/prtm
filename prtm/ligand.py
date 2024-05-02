from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformer(mol: Chem.Mol):
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
