from copy import deepcopy

import torch
from prtm.models.diffdock.datasets.process_mols import (
    generate_conformer,
    get_lig_graph_with_matching,
    moad_extract_receptor_structure,
    read_molecule,
)
from rdkit import Chem
from torch_geometric.data import Dataset, HeteroData


class InferenceDataset(Dataset):
    def __init__(
        self,
        protein_structures,
        ligands,
        lm_embeddings,
        receptor_radius=30,
        c_alpha_max_neighbors=None,
        remove_hs=False,
        all_atoms=False,
        atom_radius=5,
        atom_max_neighbors=None,
        knn_only_graph=False,
    ):
        super(InferenceDataset, self).__init__()
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.knn_only_graph = knn_only_graph

        self.ligands = ligands

        assert lm_embeddings is not None
        self.protein_structures = protein_structures
        self.lm_embeddings = lm_embeddings

    def len(self):
        return len(self.ligands)

    def get(self, idx):
        structure, ligand, lm_embedding = (
            self.protein_structures[idx],
            self.ligands[idx],
            self.lm_embeddings[idx],
        )
        structure = structure.to_protein14()

        # build the pytorch geometric heterogeneous graph
        complex_graph = HeteroData()

        # prepare the ligand
        mol = deepcopy(ligand)
        if isinstance(mol, Chem.Mol):
            Chem.SanitizeMol(mol)
            mol.RemoveAllConformers()
        else:
            mol = Chem.MolFromSmiles(mol)
            assert mol is not None
            
        mol = Chem.AddHs(mol)
        _ = generate_conformer(mol)

        get_lig_graph_with_matching(
            mol,
            complex_graph,
            popsize=None,
            maxiter=None,
            matching=False,
            keep_original=False,
            num_conformers=1,
            remove_hs=self.remove_hs,
        )

        moad_extract_receptor_structure(
            structure=structure,
            complex_graph=complex_graph,
            neighbor_cutoff=self.receptor_radius,
            max_neighbors=self.c_alpha_max_neighbors,
            lm_embeddings=lm_embedding,
            knn_only_graph=self.knn_only_graph,
            all_atoms=self.all_atoms,
            atom_cutoff=self.atom_radius,
            atom_max_neighbors=self.atom_max_neighbors,
        )

        protein_center = torch.mean(complex_graph["receptor"].pos, dim=0, keepdim=True)
        complex_graph["receptor"].pos -= protein_center
        if self.all_atoms:
            complex_graph["atom"].pos -= protein_center

        ligand_center = torch.mean(complex_graph["ligand"].pos, dim=0, keepdim=True)
        complex_graph["ligand"].pos -= ligand_center

        complex_graph.original_center = protein_center
        complex_graph.mol = mol
        complex_graph["success"] = True
        return complex_graph