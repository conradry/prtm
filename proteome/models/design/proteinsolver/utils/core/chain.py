# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""Chain class, used in Structure objects."""
import string

from .entity import Entity
from .residue import DisorderedResidue

CHAIN_IDS = list(string.ascii_uppercase + string.digits + string.ascii_lowercase)
CHAIN_IDS += [(a + b) for a in CHAIN_IDS for b in CHAIN_IDS if a != b]


class Chain(Entity):
    level = "C"

    def __lt__(self, other):
        return CHAIN_IDS.index(self.id) < CHAIN_IDS.index(other.id)

    def __le__(self, other):
        return CHAIN_IDS.index(self.id) <= CHAIN_IDS.index(other.id)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __ge__(self, other):
        return CHAIN_IDS.index(self.id) >= CHAIN_IDS.index(other.id)

    def __gt__(self, other):
        return CHAIN_IDS.index(self.id) > CHAIN_IDS.index(other.id)

    def _translate_id(self, id):
        """
        A residue id is normally a tuple (hetero flag, sequence identifier,
        insertion code). Since for most residues the hetero flag and the
        insertion code are blank (i.e. " "), you can just use the sequence
        identifier to index a residue in a chain. The _translate_id method
        translates the sequence identifier to the (" ", sequence identifier,
        " ") tuple.

        Arguments:
        o id - int, residue resseq
        """
        if isinstance(id, int):
            id = (" ", id, " ")
        return id

    # Special methods

    def __getitem__(self, id):
        """Return the residue with given id.

        The id of a residue is (hetero flag, sequence identifier, insertion code).
        If id is an int, it is translated to (" ", id, " ") by the _translate_id
        method.

        Arguments:
        o id - (string, int, string) or int
        """
        id = self._translate_id(id)
        return Entity.__getitem__(self, id)

    def __contains__(self, id):
        """True if a residue with given id is present in this chain.

        Arguments:
        o id - (string, int, string) or int
        """
        id = self._translate_id(id)
        return Entity.__contains__(self, id)

    def __delitem__(self, id):
        """
        Arguments:
        o id - (string, int, string) or int
        """
        id = self._translate_id(id)
        return Entity.__delitem__(self, id)

    def __repr__(self):
        return "<Chain id=%s>" % self.id

    # Public methods

    def get_unpacked_list(self):
        """Return a list of undisordered residues.

        Some Residue objects hide several disordered residues
        (DisorderedResidue objects). This method unpacks them,
        ie. it returns a list of simple Residue objects.
        """
        unpacked_list = []
        for residue in self:
            if isinstance(residue, DisorderedResidue):
                for dresidue in residue.disordered_get_list():
                    unpacked_list.append(dresidue)
            else:
                unpacked_list.append(residue)
        return unpacked_list

    # Public

    @property
    def residues(self):
        for r in self:
            yield r

    @property
    def atoms(self):
        for r in self:
            for a in r:
                yield a
