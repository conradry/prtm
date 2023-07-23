# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""The structure class, representing a macromolecular structure."""
from typing import List, NamedTuple

import numpy as np
import pandas as pd

from .entity import Entity


class StructureRow(NamedTuple):
    structure_id: str
    model_idx: int
    model_id: int
    chain_idx: int
    chain_id: str
    residue_idx: int
    residue_id_0: str
    residue_id_1: int
    residue_id_2: str
    residue_resname: str
    residue_segid: int
    atom_idx: int
    atom_name: str
    atom_fullname: str
    atom_x: float
    atom_y: float
    atom_z: float
    atom_bfactor: float
    atom_occupancy: float
    atom_altloc: str
    atom_serial_number: int
    #: Additional covalent bonds (like disulphide bonds).
    atom_extra_bonds: List[int]


class Structure(Entity):
    """
    The Structure class contains a collection of Model instances.
    """

    level = "S"

    def __repr__(self):
        return "<Structure id=%s>" % self.id

    def __lt__(self, other):
        return self.id.lower() < other.id.lower()

    def __le__(self, other):
        return self.id.lower() <= other.id.lower()

    def __eq__(self, other):
        return self.id.lower() == other.id.lower()

    def __ne__(self, other):
        return self.id.lower() != other.id.lower()

    def __ge__(self, other):
        return self.id.lower() >= other.id.lower()

    def __gt__(self, other):
        return self.id.lower() > other.id.lower()

    def extract_models(self, model_ids):
        # TODO: Not sure if this is neccessary
        structure = Structure(self.id)
        for model_id in model_ids:
            structure.add(self[model_id].copy())
        return structure

    def select(self, models=None, chains=None, residues=None, hetatms=None):
        """This method allows you to select things from structures using a variety of queries.

        In particular, you should be able to select one or more chains,
        and all HETATMs that are within a certain distance of those chains.
        """
        raise NotImplementedError

    def to_dataframe(self) -> pd.DataFrame:
        """Convert this structure into a pandas DataFrame."""
        data = []
        model_idx, chain_idx, residue_idx, atom_idx = -1, -1, -1, -1
        for model in self.models:
            model_idx += 1
            for chain in model.chains:
                chain_idx += 1
                for residue in chain.residues:
                    residue_idx += 1
                    for atom in residue.atoms:
                        atom_idx += 1
                        data.append(
                            StructureRow(
                                structure_id=self.id,
                                model_idx=model_idx,
                                model_id=model.id,
                                chain_idx=chain_idx,
                                chain_id=chain.id,
                                residue_idx=residue_idx,
                                residue_id_0=residue.id[0],
                                residue_id_1=residue.id[1],
                                residue_id_2=residue.id[2],
                                residue_resname=residue.resname,
                                residue_segid=residue.segid,
                                atom_idx=atom_idx,
                                atom_name=atom.name,
                                atom_fullname=atom.fullname,
                                atom_x=atom.coord[0],
                                atom_y=atom.coord[1],
                                atom_z=atom.coord[2],
                                atom_bfactor=atom.bfactor,
                                atom_occupancy=atom.occupancy,
                                atom_altloc=atom.altloc,
                                atom_serial_number=atom.serial_number,
                                atom_extra_bonds=[],
                            )
                        )
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "Structure":
        """Generate a new structure from a dataframe of atoms and an array of bonds.

        Warning:
            - If the `df` DataFrame was loaded from a CSV file using :any:`pandas.read_csv`,
              you *must* specify ``na_values=[""]`` and ``keep_default_na=False`` in order to get
              correct results. Otherwise, ``NA`` atoms may be intepreted as nulls.

        Args:
            df: DataFrame which should be converted to a Structure.

        Returns:
            structure: A :any:`Structure` object containing information present in the DataFrame.
        """
        from kmbio.PDB.core.atom import Atom
        from kmbio.PDB.core.chain import Chain
        from kmbio.PDB.core.model import Model
        from kmbio.PDB.core.residue import Residue

        assert (df["structure_id"] == df["structure_id"].iloc[0]).all()
        structure = Structure(df["structure_id"].iloc[0])
        # Groupby skips rows with NAs
        structure_df = df.drop(columns=["structure_id"])
        for (_, model_id), model_df in _groupby(
            structure_df, ["model_idx", "model_id"]
        ):
            model = Model(model_id)
            structure.add(model)
            for (_, chain_id), chain_df in _groupby(
                model_df, ["chain_idx", "chain_id"]
            ):
                chain = Chain(chain_id)
                model.add(chain)
                for (
                    (
                        _,
                        residue_id_0,
                        residue_id_1,
                        residue_id_2,
                        residue_resname,
                        residue_segid,
                    ),
                    residue_df,
                ) in _groupby(
                    chain_df,
                    [
                        "residue_idx",
                        "residue_id_0",
                        "residue_id_1",
                        "residue_id_2",
                        "residue_resname",
                        "residue_segid",
                    ],
                ):
                    residue = Residue(
                        (residue_id_0, residue_id_1, residue_id_2),
                        resname=residue_resname,
                        segid=residue_segid,
                    )
                    chain.add(residue)
                    for (_, atom_name), atom_df in _groupby(
                        residue_df, ["atom_idx", "atom_name"]
                    ):
                        assert len(atom_df) == 1
                        atom_s = atom_df.iloc[0]
                        atom = Atom(
                            name=atom_name,
                            coord=(atom_s.atom_x, atom_s.atom_y, atom_s.atom_z),
                            bfactor=atom_s.atom_bfactor,
                            occupancy=atom_s.atom_occupancy,
                            altloc=atom_s.atom_altloc,
                            fullname=atom_s.atom_fullname,
                            serial_number=atom_s.atom_serial_number,
                        )
                        residue.add(atom)
        return structure

    @property
    def models(self):
        for m in self:
            yield m

    @property
    def chains(self):
        for m in self:
            for c in m:
                yield c

    @property
    def residues(self):
        for c in self.chains:
            for r in c:
                yield r

    @property
    def atoms(self):
        for r in self.residues:
            for a in r:
                yield a


def _groupby(df, columns, *args, **kwargs):
    """Groupby columns, *not* ignoring rows containing NANs.

    Have to use this until pandas-dev/pandas#3729 is fixed.
    """
    if df[columns].isnull().any().any():
        assert not (df[columns] == np.inf).any().any()
        df[columns] = df[columns].fillna(np.inf)
        for group_key, group_df in df.groupby(columns, *args, **kwargs):
            group_key = [(k if k != np.inf else np.nan) for k in group_key]
            yield group_key, group_df
    else:
        for group_key, group_df in df.groupby(columns, *args, **kwargs):
            yield group_key, group_df
