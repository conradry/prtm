# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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


import dataclasses
import io
import re
import string
from typing import Any, Dict, List, Mapping, Optional, Self, Sequence, Union

import modelcif
import modelcif.alignment
import modelcif.dumper
import modelcif.model
import modelcif.protocol
import modelcif.qa_metric
import modelcif.reference
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure

from proteome.constants import residue_constants

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
assert PDB_MAX_CHAINS == 62

"""
Some notes about number of atoms:
- The full 37 atom representation covers all possible atoms, though no amino acid has all
of them. Trytophan has the most with 27.
- The 14 atom representation covers all atoms in the backbone and sidechain of all amino
acids excluding hydrogens. In this case the positions of the atoms are not fixed in the
array. For example, in ILE, the CD1 atom is at index 7, but in LEU it's at index 6.
- The 27 atom representation covers all atoms including hydrogens for each residue.
- The 5 atom representation covers the backbone (N, CA, C) plus O and CB.
- The 4 atom representation covers the backbone (N, CA, C) plus O.
- The 3 atom representation covers the backbone (N, CA, C).
- The 1 atom representation covers only the CA atom.
"""


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


class Protein:
    VALID_ATOM_COUNTS = [1, 3, 4, 5, 14, 27, 37]

    def __init__(
        self,
        atom_positions: Union[np.ndarray, torch.Tensor],
        aatype: Union[np.ndarray, torch.Tensor],
        atom_mask: Union[np.ndarray, torch.Tensor],
        residue_index: Union[np.ndarray, torch.Tensor],
        b_factors: Union[np.ndarray, torch.Tensor],
        chain_index: Union[np.ndarray, torch.Tensor],
        parents: Optional[Sequence[str]] = None,
        parents_chain_index: Optional[Sequence[int]] = None,
        hetatom_positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hetatom_names: Optional[Union[np.ndarray, torch.Tensor]] = None,
        remark: Optional[str] = None,
    ):
        """
        Initializes a Protein object with the given attributes. The expected number of
        atoms for this base representation is 37.

        Args:
            atom_positions (Union[np.ndarray, torch.Tensor]):
                Cartesian coordinates of atoms in angstroms. The atom types correspond to
                residue_constants.atom_types, i.e. the first three are N, CA, C. The expected
                shape is [num_res, num_atom_type, 3].
            aatype (Union[np.ndarray, torch.Tensor]):
                Amino-acid type for each residue represented as an integer between 0 and 20.
                The expected shape is [num_res].
            atom_mask (Union[np.ndarray, torch.Tensor]):
                Binary float mask to indicate presence of a particular atom. The expected
                shape is [num_res, num_atom_type].
            residue_index (Union[np.ndarray, torch.Tensor]):
                Residue index as used in PDB. The expected shape is [num_res]. The lowest index
                value is 1.
            b_factors (Union[np.ndarray, torch.Tensor]):
                B-factors, or temperature factors, of each residue. The expected shape is
                [num_res, num_atom_type].
            chain_index (Union[np.ndarray, torch.Tensor]):
                Chain indices for multi-chain predictions. The expected shape is [num_res].
            remark (Optional[str], optional):
                Optional remark about the protein. Defaults to None.
            parents (Optional[Sequence[str]], optional):
                Templates used to generate this protein. Defaults to None.
            parents_chain_index (Optional[Sequence[int]], optional):
                Chain corresponding to each parent. Defaults to None.
            hetatom_positions (Optional[Union[np.ndarray, torch.Tensor]], optional):
                HETATM positions. Defaults to None. Expected shape is [num_hetatoms, 3].
            hetatom_names (Optional[Union[np.ndarray, torch.Tensor]], optional):
                HETATM names. Defaults to None. Expected shape is [num_hetatoms].
        """
        self.atom_positions = atom_positions
        self.aatype = aatype
        self.atom_mask = atom_mask
        self.residue_index = residue_index
        self.chain_index = chain_index
        self.b_factors = b_factors
        self.parents = parents
        self.parents_chain_index = parents_chain_index
        self.hetatom_positions = hetatom_positions
        self.hetatom_names = hetatom_names
        self.remark = remark

        self.fields = [
            "atom_positions",
            "aatype",
            "atom_mask",
            "residue_index",
            "chain_index",
            "b_factors",
            "parents",
            "parents_chain_index",
            "hetatom_positions",
            "hetatom_names",
            "remark",
        ]

        self._validate_inputs()

    def _validate_inputs(self):
        """Verifies the shapes and values of the creation parameters"""
        num_res, num_atom_type = self.atom_positions.shape[:2]
        assert num_atom_type in self.VALID_ATOM_COUNTS, (
            f"Invalid number of atoms: {num_atom_type}. "
            f"Valid values are {self.VALID_ATOM_COUNTS}."
        )

        # Validate that num_res is the same for all relevant arrays
        for arr in [
            self.aatype,
            self.atom_mask,
            self.residue_index,
            self.b_factors,
            self.chain_index,
        ]:
            if arr is not None:
                assert arr.shape[0] == num_res

        # Validate that num_atom_type is the same for all relevant arrays
        for arr in [self.atom_mask, self.b_factors]:
            if arr is not None:
                assert arr.shape[1] == num_atom_type

        # Validate that the residue index is 1-indexed
        assert self.residue_index.min() == 1, "Residue index must be 1-indexed"
        assert self.chain_index.max() < PDB_MAX_CHAINS, "Chain index must be < 62"

    def to_torch(self) -> Self:
        """Converts a `Protein` instance to torch tensors."""
        prot_dict = {}
        for field in self.fields:
            v = getattr(self, field)
            if isinstance(v, np.ndarray):
                # Check if array is any float or integer type
                if np.issubdtype(v.dtype, np.floating):
                    prot_dict[field] = torch.from_numpy(v).float()
                elif np.issubdtype(v.dtype, np.integer):
                    prot_dict[field] = torch.from_numpy(v).long()
                elif np.issubdtype(v.dtype, np.bool_):
                    prot_dict[field] = torch.from_numpy(v).bool()
            elif isinstance(v, torch.tensor):
                # If already a tensor, do nothing
                prot_dict[field] = v

        return Protein(**prot_dict)

    def to_numpy(self) -> Self:
        """Converts a `Protein` instance to numpy arrays."""
        prot_dict = {}
        for field in self.fields:
            v = getattr(self, field)
            if isinstance(v, torch.Tensor):
                prot_dict[field] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                prot_dict[field] = v

        return Protein(**prot_dict)

    def get_pdb_headers(self, chain_id: int = 0) -> Sequence[str]:
        pdb_headers = []

        remark = self.remark
        if remark is not None:
            pdb_headers.append(f"REMARK {remark}")

        parents = self.parents
        parents_chain_index = self.parents_chain_index
        if parents_chain_index is not None:
            parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]

        if parents is None or len(parents) == 0:
            parents = ["N/A"]

        pdb_headers.append(f"PARENT {' '.join(parents)}")

        return pdb_headers

    def to_pdb(self) -> str:
        """Converts this `Protein` instance to a PDB string.

        Returns:
            PDB string.
        """
        restypes = residue_constants.restypes + ["X"]
        res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
        atom_types = residue_constants.atom_types

        pdb_lines = []

        atom_mask = self.atom_mask
        aatype = self.aatype
        atom_positions = self.atom_positions
        residue_index = self.residue_index.astype("int")
        b_factors = self.b_factors
        chain_index = self.chain_index

        if aatype.max() > residue_constants.restype_num:
            raise ValueError("Invalid aatypes, out of range.")

        # Construct a mapping from chain integer indices to chain ID strings.
        chain_ids = {}
        unique_fn = np.unique if isinstance(chain_index, np.ndarray) else torch.unique
        for i in unique_fn(chain_index):
            if i >= PDB_MAX_CHAINS:
                raise ValueError(
                    f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
                )
            chain_ids[i] = PDB_CHAIN_IDS[i]

        headers = self.get_pdb_headers()
        if len(headers) > 0:
            pdb_lines.extend(headers)

        pdb_lines.append("MODEL     1")
        n = aatype.shape[0]
        atom_index = 1
        last_chain_index = chain_index[0]
        prev_chain_index = 0
        chain_tags = string.ascii_uppercase

        # Add all atom sites.
        for i in range(aatype.shape[0]):
            # Close the previous chain if in a multichain PDB.
            if last_chain_index != chain_index[i]:
                pdb_lines.append(
                    _chain_end(
                        atom_index,
                        res_1to3(aatype[i - 1]),
                        chain_ids[chain_index[i - 1]],
                        residue_index[i - 1],
                    )
                )
                last_chain_index = chain_index[i]
                atom_index += 1  # Atom index increases at the TER symbol.

            res_name_3 = res_1to3(aatype[i])
            for atom_name, pos, mask, b_factor in zip(
                atom_types, atom_positions[i], atom_mask[i], b_factors[i]
            ):
                if mask < 0.5:
                    continue

                record_type = "ATOM"
                name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                element = atom_name[0]  # Protein supports only C, N, O, S, this works.
                charge = ""

                chain_tag = "A"
                if chain_index is not None:
                    chain_tag = chain_tags[chain_index[i]]

                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    # TODO: check this refactor, chose main branch version
                    # f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                    f"{res_name_3:>3} {chain_tag:>1}"
                    f"{residue_index[i]:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                pdb_lines.append(atom_line)
                atom_index += 1

            should_terminate = i == n - 1
            if chain_index is not None:
                if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                    should_terminate = True
                    prev_chain_index = chain_index[i + 1]

            if should_terminate:
                # Close the chain.
                chain_end = "TER"
                chain_termination_line = (
                    f"{chain_end:<6}{atom_index:>5}      "
                    f"{res_1to3(aatype[i]):>3} "
                    f"{chain_tag:>1}{residue_index[i]:>4}"
                )
                pdb_lines.append(chain_termination_line)
                atom_index += 1

                if i != n - 1:
                    # "prev" is a misnomer here. This happens at the beginning of
                    # each new chain.
                    pdb_lines.extend(self.get_pdb_headers(prev_chain_index))

        pdb_lines.append("ENDMDL")
        pdb_lines.append("END")

        # Pad all lines to 80 characters
        pdb_lines = [line.ljust(80) for line in pdb_lines]
        return "\n".join(pdb_lines) + "\n"  # Add terminating newline.

    def to_modelcif(self) -> str:
        """
        Converts a `Protein` instance to a ModelCIF string. Chains with identical modelled coordinates
        will be treated as the same polymer entity. But note that if chains differ in modelled regions,
        no attempt is made at identifying them as a single polymer entity.

        Args:
            prot: The protein to convert to PDB.

        Returns:
           ModelCIF string.
        """

        restypes = residue_constants.restypes + ["X"]
        atom_types = residue_constants.atom_types

        atom_mask = self.atom_mask
        aatype = self.aatype
        atom_positions = self.atom_positions
        residue_index = self.residue_index.astype("int")
        b_factors = self.b_factors
        chain_index = self.chain_index

        n = aatype.shape[0]
        if chain_index is None:
            chain_index = [0 for i in range(n)]

        system = modelcif.System(title="Model prediction")

        # Finding chains and creating entities
        seqs = {}
        seq = []
        last_chain_idx = None
        for i in range(n):
            if last_chain_idx is not None and last_chain_idx != chain_index[i]:
                seqs[last_chain_idx] = seq
                seq = []
            seq.append(restypes[aatype[i]])
            last_chain_idx = chain_index[i]
        # finally add the last chain
        seqs[last_chain_idx] = seq

        # now reduce sequences to unique ones (note this won't work if different asyms have different unmodelled regions)
        unique_seqs = {}
        for chain_idx, seq_list in seqs.items():
            seq = "".join(seq_list)
            if seq in unique_seqs:
                unique_seqs[seq].append(chain_idx)
            else:
                unique_seqs[seq] = [chain_idx]

        # adding 1 entity per unique sequence
        entities_map = {}
        for key, value in unique_seqs.items():
            model_e = modelcif.Entity(key, description="Model subunit")
            for chain_idx in value:
                entities_map[chain_idx] = model_e

        chain_tags = string.ascii_uppercase
        asym_unit_map = {}
        for chain_idx in set(chain_index):
            # Define the model assembly
            chain_id = chain_tags[chain_idx]
            asym = modelcif.AsymUnit(
                entities_map[chain_idx],
                details="Model subunit %s" % chain_id,
                id=chain_id,
            )
            asym_unit_map[chain_idx] = asym
        modeled_assembly = modelcif.Assembly(
            asym_unit_map.values(), name="Modeled assembly"
        )

        class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
            name = "pLDDT"
            software = None
            description = "Predicted lddt"

        class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
            name = "pLDDT"
            software = None
            description = "Global pLDDT, mean of per-residue pLDDTs"

        class _MyModel(modelcif.model.AbInitioModel):
            def get_atoms(self):
                # Add all atom sites.
                for i in range(n):
                    for atom_name, pos, mask, b_factor in zip(
                        atom_types, atom_positions[i], atom_mask[i], b_factors[i]
                    ):
                        if mask < 0.5:
                            continue
                        element = atom_name[
                            0
                        ]  # Protein supports only C, N, O, S, this works.
                        yield modelcif.model.Atom(
                            asym_unit=asym_unit_map[chain_index[i]],
                            type_symbol=element,
                            seq_id=residue_index[i],
                            atom_id=atom_name,
                            x=pos[0],
                            y=pos[1],
                            z=pos[2],
                            het=False,
                            biso=b_factor,
                            occupancy=1.00,
                        )

            def add_scores(self):
                # local scores
                plddt_per_residue = {}
                for i in range(n):
                    for mask, b_factor in zip(atom_mask[i], b_factors[i]):
                        if mask < 0.5:
                            continue
                        # add 1 per residue, not 1 per atom
                        if chain_index[i] not in plddt_per_residue:
                            # first time a chain index is seen: add the key and start the residue dict
                            plddt_per_residue[chain_index[i]] = {
                                residue_index[i]: b_factor
                            }
                        if residue_index[i] not in plddt_per_residue[chain_index[i]]:
                            plddt_per_residue[chain_index[i]][
                                residue_index[i]
                            ] = b_factor
                plddts = []
                for chain_idx in plddt_per_residue:
                    for residue_idx in plddt_per_residue[chain_idx]:
                        plddt = plddt_per_residue[chain_idx][residue_idx]
                        plddts.append(plddt)
                        self.qa_metrics.append(
                            _LocalPLDDT(
                                asym_unit_map[chain_idx].residue(residue_idx), plddt
                            )
                        )
                # global score
                self.qa_metrics.append((_GlobalPLDDT(np.mean(plddts))))

        # Add the model and modeling protocol to the file and write them out:
        model = _MyModel(assembly=modeled_assembly, name="Best scoring model")
        model.add_scores()

        model_group = modelcif.model.ModelGroup([model], name="All models")
        system.model_groups.append(model_group)

        fh = io.StringIO()
        modelcif.dumper.write(fh, [system])
        return fh.getvalue()

    def to_biopdb_structure(self) -> Structure:
        """Converts from a `Protein` to a BioPython PDB structure."""
        pdb_str = self.to_pdb()
        pdb_fh = io.StringIO(pdb_str)
        parser = PDBParser(QUIET=True)
        return parser.get_structure("none", pdb_fh)

    def to_rosetta_pose(self):  # can't type hint conditional import
        """Converts a protein to a PyRosetta pose."""
        try:
            from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
            from pyrosetta.rosetta.core.pose import Pose
        except:
            raise ImportError("PyRosetta is not installed")

        pose = Pose()
        pdb_str = self.to_pdb()
        pose_from_pdbstring(pose, pdb_str)
        return pose

    @classmethod
    def from_pdb_string(
        pdb_str: str,
        chain_id: Optional[str] = None,
        parse_hetatom: bool = False,
    ) -> Self:
        """Takes a PDB string and constructs a Protein object.

        WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

        Args:
            pdb_str: The contents of the pdb file
            chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
            is parsed.
            parse_hetatom: If True, then HETATM lines are parsed and returned in the hetatom_positions and hetatom_names

        Returns:
            A new `Protein` parsed from the pdb contents.
        """
        pdb_fh = io.StringIO(pdb_str)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("none", pdb_fh)
        models = list(structure.get_models())
        if len(models) != 1:
            raise ValueError(
                f"Only single model PDBs are supported. Found {len(models)} models."
            )
        model = models[0]

        atom_positions = []
        aatype = []
        atom_mask = []
        residue_index = []
        chain_ids = []
        b_factors = []

        if parse_hetatom:
            hetatom_positions = []
            hetatom_names = []

        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for res in chain:
                if res.id[2] != " ":
                    raise ValueError(
                        f"PDB contains an insertion code at chain {chain.id} and residue "
                        f"index {res.id[1]}. These are not supported."
                    )
                res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
                restype_idx = residue_constants.restype_order.get(
                    res_shortname, residue_constants.restype_num
                )

                if parse_hetatom:
                    if len(res.id[0].strip()) > 0:
                        for atom in res:
                            hetatom_positions.append(atom.coord)
                            hetatom_names.append(res.id[0].lstrip("H_"))
                        continue

                atom_count = residue_constants.atom_type_num
                pos = np.zeros((atom_count, 3))
                mask = np.zeros((atom_count,))
                res_b_factors = np.zeros((atom_count,))
                for atom in res:
                    if atom.name not in residue_constants.atom_types:
                        continue
                    else:
                        atom_index = residue_constants.atom_order[atom.name]

                    pos[atom_index] = atom.coord
                    mask[atom_index] = 1.0
                    res_b_factors[atom_index] = atom.bfactor

                if np.sum(mask) < 0.5:
                    # If no known atom positions are reported for the residue then skip it.
                    continue
                aatype.append(restype_idx)
                atom_positions.append(pos)
                atom_mask.append(mask)
                residue_index.append(res.id[1])
                chain_ids.append(chain.id)
                b_factors.append(res_b_factors)

        parents = None
        parents_chain_index = None
        if "PARENT" in pdb_str:
            parents = []
            parents_chain_index = []
            chain_id = 0
            for l in pdb_str.split("\n"):
                if "PARENT" in l:
                    if not "N/A" in l:
                        parent_names = l.split()[1:]
                        parents.extend(parent_names)
                        parents_chain_index.extend([chain_id for _ in parent_names])
                    chain_id += 1

        chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
        chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

        if parse_hetatom:
            hetatom_positions = np.array(hetatom_positions)
            hetatom_names = np.array(hetatom_names)
        else:
            hetatom_positions = None
            hetatom_names = None

        protein = Protein(
            atom_positions=np.array(atom_positions),
            atom_mask=np.array(atom_mask),
            aatype=np.array(aatype),
            residue_index=np.array(residue_index),
            chain_index=chain_index,
            b_factors=np.array(b_factors),
            parents=parents,
            parents_chain_index=parents_chain_index,
            hetatom_positions=hetatom_positions,
            hetatom_names=hetatom_names,
        )

        return protein

    @classmethod
    def from_rosetta_pose(pose):  # can't type hint conditional import
        """Converts a PyRosetta pose to a protein."""
        try:
            from pyrosetta.rosetta.std import ostringstream
        except:
            raise ImportError("PyRosetta is not installed")

        buffer = ostringstream()
        pose.dump_pdb(buffer)
        return Protein.from_pdb_string(buffer.str())


class Protein27(Protein):
    VALID_ATOM_COUNTS = [27]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Protein14(Protein):
    VALID_ATOM_COUNTS = [14]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Protein5(Protein):
    VALID_ATOM_COUNTS = [5]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Protein4(Protein):
    VALID_ATOM_COUNTS = [4]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Protein3(Protein):
    VALID_ATOM_COUNTS = [3]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ProteinCATrace(Protein):
    VALID_ATOM_COUNTS = [1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def pad_protein_14_to_27(prot: Protein, atom_pos_pad_value: float = 0.0) -> Protein:
    prot_dict = dataclasses.asdict(prot)
    prot_dict["atom_positions"] = np.pad(
        prot.atom_positions,
        ((0, 0), (0, 13), (0, 0)),
        constant_values=atom_pos_pad_value,
    )
    prot_dict["atom_mask"] = np.pad(
        prot.atom_mask, ((0, 0), (0, 13)), constant_values=0
    )
    prot_dict["b_factors"] = np.pad(prot.b_factors, ((0, 0), (0, 13)))
    prot_27_padded = Protein(**prot_dict)

    return prot_27_padded


def to_ca_only_protein(protein: Protein) -> Protein:
    """
    Strip out all atoms except CA
    """
    atom_indices = [residue_constants.atom_order["CA"]]
    return Protein(
        atom_positions=protein.atom_positions[:, atom_indices],
        atom_mask=protein.atom_mask[:, atom_indices],
        aatype=protein.aatype,
        residue_index=protein.residue_index,
        chain_index=protein.chain_index,
        b_factors=protein.b_factors[:, atom_indices],
        parents=protein.parents,
        parents_chain_index=protein.parents_chain_index,
    )


def to_backbone_only_protein(protein: Protein) -> Protein:
    """
    Strip out all atoms except those in the backbone [N, CA, C, O]
    """
    atom_indices = [
        residue_constants.atom_order[atom] for atom in ["N", "CA", "C", "O"]
    ]
    return Protein(
        atom_positions=protein.atom_positions[:, atom_indices],
        atom_mask=protein.atom_mask[:, atom_indices],
        aatype=protein.aatype,
        residue_index=protein.residue_index,
        chain_index=protein.chain_index,
        b_factors=protein.b_factors[:, atom_indices],
        parents=protein.parents,
        parents_chain_index=protein.parents_chain_index,
    )


def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling
    """
    out_pdb_lines = []
    lines = pdb_str.split("\n")

    remark = prot.remark
    if remark is not None:
        out_pdb_lines.append(f"REMARK {remark}")

    parents_per_chain = None
    if prot.parents is not None and len(prot.parents) > 0:
        parents_per_chain = []
        if prot.parents_chain_index is not None:
            cur_chain = prot.parents_chain_index[0]
            parent_dict = {}
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)

            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ["N/A"])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(prot.parents)
    else:
        parents_per_chain = [["N/A"]]

    make_parent_line = lambda p: f"PARENT {' '.join(p)}"

    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))

    chain_counter = 0
    for i, l in enumerate(lines):
        if "PARENT" not in l and "REMARK" not in l:
            out_pdb_lines.append(l)
        if "TER" in l and not "END" in lines[i + 1]:
            chain_counter += 1
            if not chain_counter >= len(parents_per_chain):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ["N/A"]

            out_pdb_lines.append(make_parent_line(chain_parents))

    return "\n".join(out_pdb_lines)


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def add_oxygen_to_atom_positions(atom_positions: np.ndarray) -> np.ndarray:
    """Adds oxygen atoms to the atom positions.

    Args:
        atom_positions: A numpy array of shape [N, 3, 3] where N is the number of
            atoms in the protein. The last dimension is the x, y, z coordinates
            of the atoms.
    Returns:
        atom_positions_with_oxygen: A numpy array of shape [N, 4, 3].
    """
    assert atom_positions.shape[-1] == 3
    # Constants for oxygen placement
    L = 1.231
    A = 2.108
    D = -3.142

    # Unpacking
    N = np.roll(atom_positions[:, 0, :], -1, axis=0)
    CA = atom_positions[:, 1, :]
    C = atom_positions[:, 2, :]

    unit_norm = lambda x: x / (1e-5 + np.linalg.norm(x, axis=1, keepdims=True))
    bc = unit_norm(CA - C)
    n = unit_norm(np.cross(CA - N, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]

    O = C + sum([m * d for m, d in zip(m, d)])
    return np.concatenate((atom_positions, O[:, None, :]), axis=1)
