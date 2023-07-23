# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""Residue class, used by Structure objects."""

from .atom import DisorderedAtom
from .entity import DisorderedEntityWrapper, Entity

_atom_name_dict = {}
_atom_name_dict["N"] = 1
_atom_name_dict["CA"] = 2
_atom_name_dict["C"] = 3
_atom_name_dict["O"] = 4


class Residue(Entity):
    """
    Represents a residue. A Residue object stores atoms.
    """

    level = "R"

    def __init__(self, id, resname, segid, **kwargs):
        self.disordered = 0
        self.resname = resname
        self.segid = segid
        super().__init__(id, **kwargs)

    def __repr__(self):
        resname = self.resname
        hetflag, resseq, icode = self.id
        full_id = (resname, hetflag, resseq, icode)
        return "<Residue %s het=%s resseq=%s icode=%s>" % full_id

    def __lt__(self, other):
        return self.id[1] < other.id[1]

    def __le__(self, other):
        return self.id[1] <= other.id[1]

    def __eq__(self, other):
        return self.id == other.id and self.resname == other.resname

    def __ne__(self, other):
        return self.id != other.id or self.resname != other.resname

    def __ge__(self, other):
        return self.id[1] >= other.id[1]

    def __gt__(self, other):
        return self.id[1] > other.id[1]

    def get_unpacked_list(self):
        """Returns the list of all atoms, unpack DisorderedAtoms."""
        undisordered_atom_list = []
        for atom in self:
            if isinstance(atom, DisorderedAtom):
                undisordered_atom_list = (
                    undisordered_atom_list + atom.disordered_get_list()
                )
            else:
                undisordered_atom_list.append(atom)
        return undisordered_atom_list

    @property
    def is_hetatm(self):
        bool(self.id[0].strip())

    @property
    def atoms(self):
        for a in self:
            yield a


class DisorderedResidue(DisorderedEntityWrapper):
    """
    DisorderedResidue is a wrapper around two or more Residue objects. It is
    used to represent point mutations (e.g. there is a Ser 60 and a Cys 60 residue,
    each with 50 % occupancy).
    """

    def __init__(self, id):
        DisorderedEntityWrapper.__init__(self, id)

    def __repr__(self):
        resname = self.resname
        hetflag, resseq, icode = self.id
        full_id = (resname, hetflag, resseq, icode)
        return "<DisorderedResidue %s het=%s resseq=%i icode=%s>" % full_id

    def add(self, atom):
        residue = self.disordered_get()
        if not isinstance(atom, DisorderedAtom):
            # Atoms in disordered residues should have non-blank
            # altlocs, and are thus represented by DisorderedAtom objects.
            resname = residue.resname
            het, resseq, icode = residue.id
            # add atom anyway, if PDBParser ignores exception the atom will be part of the residue
            residue.add(atom)
            raise Exception(
                "Blank altlocs in duplicate residue %s (%s, %i, %s)"
                % (resname, het, resseq, icode)
            )
        residue.add(atom)

    def sort(self):
        "Sort the atoms in the child Residue objects."
        for residue in self.disordered_get_list():
            residue.sort()

    def disordered_add(self, residue):
        """Add a residue object and use its resname as key.

        Arguments:
        o residue - Residue object
        """
        resname = residue.resname
        # add chain parent to residue
        chain = self.parent
        residue.parent = chain
        assert not self.disordered_has_id(resname)
        self[resname] = residue
        self.disordered_select(resname)
