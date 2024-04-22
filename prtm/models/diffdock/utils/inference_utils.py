import os

import torch
from Bio.PDB import PDBParser
from prtm.models.diffdock.datasets.constants import three_to_one
from prtm.models.diffdock.datasets.process_mols import (
    generate_conformer,
    get_lig_graph_with_matching,
    moad_extract_receptor_structure,
    read_molecule,
)
from rdkit.Chem import AddHs, MolFromSmiles
from torch_geometric.data import Dataset, HeteroData


def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure("random_id", file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq = ""
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
            if (
                c_alpha != None and n != None and c != None
            ):  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += "-"
                    print(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex. Replacing it with a dash - .",
                    )

        if sequence is None:
            sequence = seq
        else:
            sequence += ":" + seq

    return sequence


def set_nones(l):
    return [s if str(s) != "nan" else None for s in l]


def get_sequences(protein_files, protein_sequences):
    new_sequences = []
    for i in range(len(protein_files)):
        if protein_files[i] is not None:
            new_sequences.append(get_sequences_from_pdbfile(protein_files[i]))
        else:
            new_sequences.append(protein_sequences[i])
    return new_sequences


def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory on chunk_size", chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


class InferenceDataset(Dataset):
    def __init__(
        self,
        out_dir,
        complex_names,
        protein_files,
        ligand_descriptions,
        protein_sequences,
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

        self.complex_names = complex_names
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences

        assert None not in protein_files
        assert lm_embeddings is not None
        self.protein_files = protein_files
        self.lm_embeddings = lm_embeddings

    def len(self):
        return len(self.complex_names)

    def get(self, idx):
        name, protein_file, ligand_description, lm_embedding = (
            self.complex_names[idx],
            self.protein_files[idx],
            self.ligand_descriptions[idx],
            self.lm_embeddings[idx],
        )

        # build the pytorch geometric heterogeneous graph
        complex_graph = HeteroData()
        complex_graph["name"] = name

        # parse the ligand, either from file or smile
        try:
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path

            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    raise Exception(
                        "RDKit could not read the molecule ", ligand_description
                    )
                mol.RemoveAllConformers()
                mol = AddHs(mol)
                generate_conformer(mol)
        except Exception as e:
            print(
                "Failed to read molecule ",
                ligand_description,
                " We are skipping it. The reason is the exception: ",
                e,
            )
            complex_graph["success"] = False
            return complex_graph

        # try:
        # parse the receptor from the pdb file
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
            path=os.path.join(protein_file),
            complex_graph=complex_graph,
            neighbor_cutoff=self.receptor_radius,
            max_neighbors=self.c_alpha_max_neighbors,
            lm_embeddings=lm_embedding,
            knn_only_graph=self.knn_only_graph,
            all_atoms=self.all_atoms,
            atom_cutoff=self.atom_radius,
            atom_max_neighbors=self.atom_max_neighbors,
        )

        # except Exception as e:
        #    print(f"Skipping {name} because of the error:")
        #    print(e)
        #    complex_graph["success"] = False
        #    return complex_graph

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
