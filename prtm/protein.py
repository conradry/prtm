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


from __future__ import annotations

import io
import os
import string
from collections import namedtuple
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import modelcif
import modelcif.alignment
import modelcif.dumper
import modelcif.model
import modelcif.protocol
import modelcif.qa_metric
import modelcif.reference
import numpy as np
import torch
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from prtm.constants import residue_constants
from prtm.visual import view_ca_trace, view_protein_with_bfactors

try:
    import pyrosetta
    from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
    from pyrosetta.rosetta.core.pose import Pose

    pyrosetta.init(silent=True, extra_options="-mute all")
except:
    pass

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
TODO: Add validation to check that the atom mask for this representation is a subset of
the ideal_atom_mask. If it's not that means there is some funny business.
- The 27 atom representation covers all atoms including hydrogens for each residue.
NOTE: The parser currently ignore hydrogen atoms in the PDB file so this representation
is currently always equivalent to the 14 atom representation but with zero padding. Actually
the order won't match the convention for hydrogens in the PDB format so be careful with defining
this one. It may be better to pad Protein14 to shape 27 while ignoring the hydrogens or to
create a new atom ordering that includes the hydrogens after all other atoms.
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


def get_structure_from_pdb(pdb_id: str) -> str:
    """Downloads and reads a pdb file from the RCSB database."""
    pdb_path = os.path.expanduser("~/.prtm/pdb_downloads/")
    _ = PDB.PDBList().retrieve_pdb_file(
        pdb_code=pdb_id, pdir=pdb_path, file_format="pdb"
    )
    file_path = os.path.join(pdb_path, f"pdb{pdb_id.lower()}.ent")
    with open(file_path, mode="r") as f:
        pdb_str = f.read()

    return pdb_str


def parse_pdb_string(
    pdb_str: str,
    chain_id: Optional[str] = None,
    parse_hetatom: bool = True,
) -> Dict[str, np.ndarray]:
    """Takes a PDB string and parses it into arrays.

    WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file
        chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
        is parsed.
        parse_hetatom: If True, then HETATM lines are parsed and returned in the hetatom_positions and hetatom_names

    Returns:
        A dictionary with attributes to construct a protein class
        and arrays from the parsed pdb.
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
            else:
                if len(res.id[0].strip()) > 0:
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

    protein_dict = dict(
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

    return protein_dict


def superimpose_structures(
    target_structure: ProteinBase,
    moving_structure: ProteinBase,
):
    """Superimpose two structures using the CA atoms.

    Args:
        target_structure: The structure to superimpose onto.
        moving_structure: The structure to superimpose onto the target structure.

    Returns:
        A new `Protein` instance with the moving structure superimposed onto the
        target structure.
    """
    # Must have the same sequence in order to superimpose
    assert len(target_structure.sequence()) == len(
        moving_structure.sequence()
    ), "Sequences must be the same to superimpose structures."

    # Only 1 model per protein by construction
    target_model = target_structure.to_biopdb_structure()[0]
    moving_model = moving_structure.to_biopdb_structure()[0]

    # Get lists of all the CA atoms
    target_atoms = []
    moving_atoms = []
    for target_chain, moving_chain in zip(target_model, moving_model):
        for target_res, moving_res in zip(target_chain, moving_chain):
            target_atoms.append(target_res["CA"])
            moving_atoms.append(moving_res["CA"])

    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(target_atoms, moving_atoms)
    super_imposer.apply(moving_model.get_atoms())

    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(moving_model)

    # Write the pdb str to an io string
    pdb_fh = io.StringIO()
    pdb_io.save(pdb_fh)

    return type(moving_structure).from_pdb_string(pdb_fh.getvalue())


class ProteinBase:
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

        assert self.chain_index.max() < PDB_MAX_CHAINS, "Chain index must be < 62"

    def to_torch(self) -> ProteinBase:
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
                else:
                    prot_dict[field] = v
            elif isinstance(v, torch.Tensor):
                # If already a tensor, do nothing
                prot_dict[field] = v

        return type(self)(**prot_dict)

    def to_numpy(self) -> ProteinBase:
        """Converts a `Protein` instance to numpy arrays."""
        prot_dict = {}
        for field in self.fields:
            v = getattr(self, field)
            if isinstance(v, torch.Tensor):
                prot_dict[field] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                prot_dict[field] = v

        return type(self)(**prot_dict)

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

    def _to_pdb_from_atom37(self) -> str:
        """Converts this `Protein` instance to a PDB string.
        This is a private method because children should have
        an appropriate to_pdb method that makes sure there are 37
        atoms (or a compatible subset) before calling this method.

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

    def to_pdb(self):
        raise NotImplementedError

    def _to_modelcif_from_atom37(self) -> str:
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

    def to_modelcif(self):
        raise NotImplementedError

    def to_biopdb_structure(self) -> Structure:
        """Converts from a `Protein` to a BioPython PDB structure."""
        pdb_str = self.to_pdb()
        pdb_fh = io.StringIO(pdb_str)
        parser = PDBParser(QUIET=True)
        return parser.get_structure("none", pdb_fh)

    def to_rosetta_pose(self):  # can't type hint conditional import
        """Converts a protein to a PyRosetta pose."""
        if "pyrosetta" not in globals():
            raise Exception("PyRosetta is not imported.")

        pose = Pose()
        pdb_str = self.to_pdb()
        pose_from_pdbstring(pose, pdb_str)
        return pose

    def to_dict(self):
        # Export all the fields into a dict
        return {field: getattr(self, field) for field in self.fields}

    @classmethod
    def from_pdb_string(
        cls,
        pdb_str: str,
        chain_id: Optional[str] = None,
        parse_hetatom: bool = True,
    ) -> ProteinBase:
        raise NotImplementedError

    @classmethod
    def from_rosetta_pose(cls, pose):  # can't type hint conditional import
        """Converts a PyRosetta pose to a protein."""
        try:
            from pyrosetta.rosetta.std import ostringstream
        except:
            raise ImportError("PyRosetta is not installed")

        buffer = ostringstream()
        pose.dump_pdb(buffer)
        return cls.from_pdb_string(buffer.str())

    def _pad_to_n_atoms(self, n: int) -> Dict[str, np.ndarray]:
        # Trivially pad such that relevant arrays have n atoms
        protein: ProteinBase = self.to_numpy()
        num_atoms = protein.atom_positions.shape[1]
        atom_positions = np.pad(
            protein.atom_positions,
            ((0, 0), (0, n - num_atoms), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        atom_mask = np.pad(
            protein.atom_mask,
            ((0, 0), (0, n - num_atoms)),
            mode="constant",
            constant_values=0.0,
        )
        b_factors = np.pad(
            protein.b_factors,
            ((0, 0), (0, n - num_atoms)),
            mode="constant",
            constant_values=0.0,
        )

        return dict(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=protein.aatype,
            residue_index=protein.residue_index,
            chain_index=protein.chain_index,
            b_factors=b_factors,
            parents=protein.parents,
            parents_chain_index=protein.parents_chain_index,
            remark=protein.remark,
            hetatom_positions=protein.hetatom_positions,
            hetatom_names=protein.hetatom_names,
        )

    def _crop_n_atoms(self, n: int) -> Dict[str, np.ndarray]:
        # Trivially crop out the first n atoms
        protein: ProteinBase = self.to_numpy()
        atom_positions = protein.atom_positions[:, :n]
        atom_mask = protein.atom_mask[:, :n]
        b_factors = protein.b_factors[:, :n]

        return dict(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=protein.aatype,
            residue_index=protein.residue_index,
            chain_index=protein.chain_index,
            b_factors=b_factors,
            parents=protein.parents,
            parents_chain_index=protein.parents_chain_index,
            remark=protein.remark,
            hetatom_positions=protein.hetatom_positions,
            hetatom_names=protein.hetatom_names,
        )

    def to_protein37(self) -> Protein37:
        raise NotImplementedError

    def to_protein27(self) -> Protein27:
        raise NotImplementedError

    def to_protein14(self) -> Protein14:
        raise NotImplementedError

    def to_protein5(self) -> Protein5:
        return Protein5(**self._crop_n_atoms(n=5))

    def to_protein4(self) -> Protein4:
        return Protein4(**self._crop_n_atoms(n=4))

    def to_protein3(self) -> Protein3:
        return Protein3(**self._crop_n_atoms(n=3))

    def to_ca_trace(self) -> ProteinCATrace:
        # Trim out the second atom which is CA
        atom_positions = self.atom_positions[:, [1]]
        atom_mask = self.atom_mask[:, [1]]
        b_factors = self.b_factors[:, [1]]

        return ProteinCATrace(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=self.aatype,
            residue_index=self.residue_index,
            chain_index=self.chain_index,
            b_factors=b_factors,
            parents=self.parents,
            parents_chain_index=self.parents_chain_index,
            remark=self.remark,
            hetatom_positions=self.hetatom_positions,
            hetatom_names=self.hetatom_names,
        )

    def get_chain(self, chain_id: str) -> ProteinBase:
        assert chain_id in PDB_CHAIN_IDS, f"Invalid chain_id: {chain_id}"
        chain_index = PDB_CHAIN_IDS.index(chain_id)
        # Get the mask for indices in this chain
        chain_mask = self.chain_index == chain_index
        # Create a new Protein instance that only includes the
        # relevant chain
        return type(self)(
            atom_positions=self.atom_positions[chain_mask],
            atom_mask=self.atom_mask[chain_mask],
            aatype=self.aatype[chain_mask],
            residue_index=self.residue_index[chain_mask],
            chain_index=self.chain_index[chain_mask],
            b_factors=self.b_factors[chain_mask],
            parents=self.parents,
            parents_chain_index=self.parents_chain_index,
            remark=self.remark,
            hetatom_positions=self.hetatom_positions,
            hetatom_names=self.hetatom_names,
        )

    def sequence(self, chain_id: Optional[str] = None) -> str:
        # Decode the aatype sequence to a string
        if chain_id is not None:
            assert chain_id in PDB_CHAIN_IDS, f"Invalid chain_id: {chain_id}"
            aatypes = self.aatype[self.chain_index == PDB_CHAIN_IDS.index(chain_id)]
        else:
            aatypes = self.aatype

        return "".join(residue_constants.restypes[a] for a in aatypes)

    def show(
        self,
        cmap: str = "viridis",
        bfactor_is_confidence: bool = True,
        show_sidechains: bool = True,
    ):
        return view_protein_with_bfactors(
            self,
            cmap=cmap,
            bfactor_is_confidence=bfactor_is_confidence,
            show_sidechains=show_sidechains,
        )

    @property
    def shape(self):
        # Get the shape as a named tuple of (num_residues, num_atoms, num_chains)
        shape_tuple = namedtuple(
            "ProteinShape", ["num_residues", "num_atoms", "num_chains"]
        )
        return shape_tuple(
            num_residues=self.aatype.shape[0],
            num_atoms=self.atom_positions.shape[1],
            num_chains=len(np.unique(self.chain_index)),
        )

    @property
    def chains(self):
        """Returns a string with all available chains."""
        return "".join(PDB_CHAIN_IDS[i] for i in np.unique(self.chain_index))

    def superimpose(self, other: ProteinBase) -> ProteinBase:
        """Superimposes another protein onto this protein."""
        return superimpose_structures(self, other)


class Protein37(ProteinBase):
    VALID_ATOM_COUNTS = [37]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_pdb(self) -> str:
        return self._to_pdb_from_atom37()

    def to_modelcif(self) -> str:
        return self._to_modelcif_from_atom37()

    def to_protein14(self) -> Protein14:
        mapping = residue_constants.restype_atom14_to_atom37
        protein37: Protein37 = self.to_numpy()
        # Get the indices of the atoms to keep from aatype
        atom_indices = mapping[protein37.aatype]
        residue_mask = ~(residue_constants.restype2atom14_mask > 0)
        atom14_mask = residue_mask[protein37.aatype]

        seq_len = protein37.aatype.shape[0]
        row_indices = np.arange(seq_len)[:, None]

        # Extract atoms from all relevant arrays
        atom_positions = protein37.atom_positions[row_indices, atom_indices]
        atom_mask = protein37.atom_mask[row_indices, atom_indices]
        b_factors = protein37.b_factors[row_indices, atom_indices]

        # Zero out the atoms that are not present in the new representation
        atom_positions[atom14_mask] = 0.0
        atom_mask[atom14_mask] = 0.0
        b_factors[atom14_mask] = 0.0

        return Protein14(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=protein37.aatype,
            residue_index=protein37.residue_index,
            chain_index=protein37.chain_index,
            b_factors=b_factors,
            parents=protein37.parents,
            parents_chain_index=protein37.parents_chain_index,
            remark=protein37.remark,
            hetatom_positions=protein37.hetatom_positions,
            hetatom_names=protein37.hetatom_names,
        )

    def to_protein27(self) -> Protein27:
        protein14: Protein14 = self.to_protein14()
        return protein14.to_protein27()

    def to_protein37(self) -> Protein37:
        return self

    @classmethod
    def from_pdb_string(
        cls,
        pdb_str: str,
        chain_id: Optional[str] = None,
        parse_hetatom: bool = True,
    ) -> ProteinBase:
        """Converts a PDB string to a protein.

        Args:
            pdb_str: The contents of the pdb file
            chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
            is parsed.
            parse_hetatom: If True, then HETATM lines are parsed and returned in the hetatom_positions and hetatom_names

        Returns:
            A `Protein` instance.
        """
        prot_dict = parse_pdb_string(pdb_str, chain_id, parse_hetatom)
        return cls(**prot_dict)


class Protein14(ProteinBase):
    VALID_ATOM_COUNTS = [14]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_protein37(self) -> Protein37:
        mapping = residue_constants.restype_atom37_to_atom14
        protein14: Protein14 = self.to_numpy()
        # Get the indices of the atoms to keep from aatype
        atom_indices = mapping[protein14.aatype]
        residue_mask = ~(residue_constants.STANDARD_ATOM_MASK > 0)
        atom37_mask = residue_mask[protein14.aatype]

        seq_len = protein14.aatype.shape[0]
        row_indices = np.arange(seq_len)[:, None]

        # Extract atoms from all relevant arrays
        atom_positions = protein14.atom_positions[row_indices, atom_indices]
        atom_mask = protein14.atom_mask[row_indices, atom_indices]
        b_factors = protein14.b_factors[row_indices, atom_indices]

        # Zero out the atoms that are not present in the new representation
        atom_positions[atom37_mask] = 0.0
        atom_mask[atom37_mask] = 0.0
        b_factors[atom37_mask] = 0.0

        return Protein37(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=protein14.aatype,
            residue_index=protein14.residue_index,
            chain_index=protein14.chain_index,
            b_factors=b_factors,
            parents=protein14.parents,
            parents_chain_index=protein14.parents_chain_index,
            remark=protein14.remark,
            hetatom_positions=protein14.hetatom_positions,
            hetatom_names=protein14.hetatom_names,
        )

    def to_protein27(self) -> Protein27:
        return Protein27(**self._pad_to_n_atoms(n=27))

    def to_protein14(self) -> Protein14:
        return self

    def to_pdb(self) -> str:
        protein37 = self.to_protein37()
        return protein37.to_pdb()

    def to_modelcif(self) -> str:
        protein37 = self.to_protein37()
        return protein37.to_modelcif()

    @classmethod
    def from_pdb_string(
        cls,
        pdb_str: str,
        chain_id: Optional[str] = None,
        parse_hetatom: bool = True,
    ) -> ProteinBase:
        """Converts a PDB string to a protein.

        Args:
            pdb_str: The contents of the pdb file
            chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
            is parsed.
            parse_hetatom: If True, then HETATM lines are parsed and returned in the hetatom_positions and hetatom_names

        Returns:
            A `Protein` instance.
        """
        prot_dict = parse_pdb_string(pdb_str, chain_id, parse_hetatom)
        protein37 = Protein37(**prot_dict)
        return protein37.to_protein14()


class Protein27(ProteinBase):
    VALID_ATOM_COUNTS = [27]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_protein14(self) -> Protein14:
        # Trivially crop out the first 14 atoms
        return Protein14(**self._crop_n_atoms(n=14))

    def to_protein37(self) -> Protein37:
        protein14 = self.to_protein14()
        return protein14.to_protein37()

    def to_protein27(self) -> Protein27:
        return self

    def to_pdb(self) -> str:
        protein37 = self.to_protein37()
        return protein37.to_pdb()

    def to_modelcif(self) -> str:
        protein37 = self.to_protein37()
        return protein37.to_modelcif()

    @classmethod
    def from_pdb_string(
        cls,
        pdb_str: str,
        chain_id: Optional[str] = None,
        parse_hetatom: bool = True,
    ) -> ProteinBase:
        """Converts a PDB string to a protein.

        Args:
            pdb_str: The contents of the pdb file
            chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
            is parsed.
            parse_hetatom: If True, then HETATM lines are parsed and returned in the hetatom_positions and hetatom_names

        Returns:
            A `Protein` instance.
        """
        prot_dict = parse_pdb_string(pdb_str, chain_id, parse_hetatom)
        protein37 = Protein37(**prot_dict)
        return protein37.to_protein27()


class Protein5(ProteinBase):
    VALID_ATOM_COUNTS = [5]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_protein14(self) -> Protein14:
        # Trivially pad such that relevant arrays have 14 atoms
        return Protein14(**self._pad_to_n_atoms(n=14))

    def to_protein27(self) -> Protein27:
        # Trivially pad such that relevant arrays have 27 atoms
        return Protein27(**self._pad_to_n_atoms(n=27))

    def to_protein37(self) -> Protein37:
        # Trivially pad such that relevant arrays have 37 atoms
        return Protein37(**self._pad_to_n_atoms(n=37))

    def to_protein5(self) -> Protein5:
        return self

    def to_pdb(self) -> str:
        protein37 = self.to_protein37()
        return protein37.to_pdb()

    def to_modelcif(self) -> str:
        protein37 = self.to_protein37()
        return protein37.to_modelcif()

    @classmethod
    def from_pdb_string(
        cls,
        pdb_str: str,
        chain_id: Optional[str] = None,
        parse_hetatom: bool = True,
    ) -> ProteinBase:
        """Converts a PDB string to a protein.

        Args:
            pdb_str: The contents of the pdb file
            chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
            is parsed.
            parse_hetatom: If True, then HETATM lines are parsed and returned in the hetatom_positions and hetatom_names

        Returns:
            A `Protein` instance.
        """
        prot_dict = parse_pdb_string(pdb_str, chain_id, parse_hetatom)
        protein37 = Protein37(**prot_dict)
        return protein37.to_protein5()


class Protein4(Protein5):
    VALID_ATOM_COUNTS = [4]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_protein4(self) -> Protein4:
        return self


class Protein3(Protein5):
    VALID_ATOM_COUNTS = [3]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_protein3(self) -> Protein3:
        return self


class ProteinCATrace(ProteinBase):
    VALID_ATOM_COUNTS = [1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_protein3(self):
        # The CA atom is the second of 3 atoms so we
        # need to pad with a zero first atom and zero last
        # atom
        protein_ca: ProteinCATrace = self.to_numpy()
        atom_positions = np.pad(
            protein_ca.atom_positions,
            ((0, 0), (1, 1), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        atom_mask = np.pad(
            protein_ca.atom_mask,
            ((0, 0), (1, 1)),
            mode="constant",
            constant_values=0.0,
        )
        b_factors = np.pad(
            protein_ca.b_factors,
            ((0, 0), (1, 1)),
            mode="constant",
            constant_values=0.0,
        )
        return Protein3(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=protein_ca.aatype,
            residue_index=protein_ca.residue_index,
            chain_index=protein_ca.chain_index,
            b_factors=b_factors,
            parents=protein_ca.parents,
            parents_chain_index=protein_ca.parents_chain_index,
            remark=protein_ca.remark,
            hetatom_positions=protein_ca.hetatom_positions,
            hetatom_names=protein_ca.hetatom_names,
        )

    def to_pdb(self) -> str:
        protein3 = self.to_protein3()
        return protein3.to_pdb()

    def to_modelcif(self) -> str:
        protein3 = self.to_protein3()
        return protein3.to_modelcif()

    def to_protein14(self) -> Protein14:
        protein3 = self.to_protein3()
        return protein3.to_protein14()

    def to_protein27(self) -> Protein27:
        protein3 = self.to_protein3()
        return protein3.to_protein27()

    def to_protein37(self) -> Protein37:
        protein3 = self.to_protein3()
        return protein3.to_protein37()

    def to_protein5(self) -> Protein5:
        protein3 = self.to_protein3()
        return protein3.to_protein5()

    def to_protein4(self) -> Protein4:
        protein3 = self.to_protein3()
        return protein3.to_protein4()

    def to_ca_trace(self) -> ProteinCATrace:
        return self

    @classmethod
    def from_pdb_string(
        cls,
        pdb_str: str,
        chain_id: Optional[str] = None,
        parse_hetatom: bool = True,
    ) -> ProteinBase:
        """Converts a PDB string to a protein.

        Args:
            pdb_str: The contents of the pdb file
            chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
            is parsed.
            parse_hetatom: If True, then HETATM lines are parsed and returned in the hetatom_positions and hetatom_names

        Returns:
            A `Protein` instance.
        """
        prot_dict = parse_pdb_string(pdb_str, chain_id, parse_hetatom)
        protein37 = Protein37(**prot_dict)
        return protein37.to_ca_trace()

    def show(
        self,
        cmap: str = "blue",
        bfactor_is_confidence: bool = True,
        show_sidechains: bool = False,
    ):
        return view_ca_trace(self, color=cmap)


def add_pdb_headers(prot: ProteinBase, pdb_str: str) -> str:
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


def ideal_atom37_mask(aatype: np.ndarray) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[aatype]


def ideal_atom27_mask(aatype: np.ndarray) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.restype2atom27_mask[aatype]


def ideal_atom14_mask(aatype: np.ndarray) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.restype2atom14_mask[aatype]


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
