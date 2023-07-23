# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""Atom class, used in Structure objects."""

import copy
import logging

import numpy as np
from Bio.Data import IUPACData

from .entity import DisorderedEntityWrapper, Entity
from .vector import Vector

logger = logging.getLogger(__name__)


class Atom(Entity):
    level = "A"

    def __init__(
        self,
        name,
        coord,
        bfactor,
        occupancy,
        altloc,
        fullname,
        serial_number,
        element=None,
        **kwargs,
    ):
        """Create Atom object.

        The Atom object stores atom name (both with and without spaces),
        coordinates, B factor, occupancy, alternative location specifier
        and (optionally) anisotropic B factor and standard deviations of
        B factor and positions.

        @param name: atom name (eg. "CA"). Note that spaces are normally stripped.
        @type name: string

        @param coord: atomic coordinates (x,y,z)
        @type coord: Numeric array (Float0, size 3)

        @param bfactor: isotropic B factor
        @type bfactor: number

        @param occupancy: occupancy (0.0-1.0)
        @type occupancy: number

        @param altloc: alternative location specifier for disordered atoms
        @type altloc: string

        @param fullname: full atom name, including spaces, e.g. " CA ". Normally
        these spaces are stripped from the atom name.
        @type fullname: string

        @param element: atom element, e.g. "C" for Carbon, "HG" for mercury,
        @type element: uppercase string (or None if unknown)
        """
        super().__init__(name, **kwargs)
        # Reference to the residue
        self.parent = None
        # the atomic data
        self.name = name  # eg. CA, spaces are removed from atom name
        self.fullname = fullname  # e.g. " CA ", spaces included
        self.coord = coord
        self.bfactor = bfactor
        self.occupancy = occupancy
        self.altloc = altloc
        self._full_id = None  # (structure id, model id, chain id, residue id, atom id)
        self._id = name  # id of atom is the atom name (e.g. "CA")
        self.disordered = 0
        self.anisou_array = None
        self.siguij_array = None
        self.sigatm_array = None
        self.serial_number = serial_number
        # Dictionary that keeps additional properties
        assert not element or element == element.upper(), element
        self.element = self._assign_element(element)
        self.mass = self._assign_atom_mass()

    def _assign_element(self, element):
        """Tries to guess element from atom name if not recognised."""
        if not element or element.capitalize() not in IUPACData.atom_weights:
            # Inorganic elements have their name shifted left by one position
            #  (is a convention in PDB, but not part of the standard).
            # isdigit() check on last two characters to avoid mis-assignment of
            # hydrogens atoms (GLN HE21 for example)

            if self.fullname[0].isalpha() and not self.fullname[2:].isdigit():
                putative_element = self.name.strip()
            else:
                # Hs may have digit in [0]
                if self.name[0].isdigit():
                    putative_element = self.name[1]
                else:
                    putative_element = self.name[0]

            if putative_element.capitalize() in IUPACData.atom_weights:
                msg = "Used element %r for Atom (name=%s) with given element %r" % (
                    putative_element,
                    self.name,
                    element,
                )
                element = putative_element
            else:
                msg = (
                    "Could not assign element %r for Atom (name=%s) with given element %r"
                    % (
                        putative_element,
                        self.name,
                        element,
                    )
                )
                element = ""
            logger.info(msg)

        return element

    def _assign_atom_mass(self):
        # Needed for Bio/Struct/Geometry.py C.O.M. function
        if self.element:
            return IUPACData.atom_weights[self.element.capitalize()]
        else:
            return float("NaN")

    def __repr__(self):
        """Print Atom object as <Atom atom_name>."""
        return "<Atom %s>" % self.id

    def __sub__(self, other):
        """Calculate distance between two atoms.

        Example:
            >>> distance=atom1-atom2

        @param other: the other atom
        @type other: L{Atom}
        """
        diff = self.coord - other.coord
        return np.sqrt(np.dot(diff, diff))

    def __eq__(self, other):
        return self.atoms_equal(other)

    def __ne__(self, other):
        return not self.atoms_equal(other)

    def atoms_equal(self, other, atol=1e-3):
        """Check whether two atoms are equal.

        Unlike `__eq__` and `__ne__` special methods, `atoms_equal` allows you
        to specify tolerance.
        """
        return self.name == other.name and np.allclose(
            self.coord, other.coord, atol=atol
        )

    @property
    def full_id(self):
        """Return the full id of the atom.

        The full id of an atom is the tuple
        (structure id, model id, chain id, residue id, atom name, altloc).
        """
        return self.parent.full_id + ((self.name, self.altloc),)

    def transform(self, rot, tran):
        """Apply rotation and translation to the atomic coordinates.

        Example:
                >>> rotation=rotmat(pi, Vector(1, 0, 0))
                >>> translation=array((0, 0, 1))
                >>> atom.transform(rotation, translation)

        @param rot: A right multiplying rotation matrix
        @type rot: 3x3 Numeric array

        @param tran: the translation vector
        @type tran: size 3 Numeric array
        """
        self.coord = np.dot(self.coord, rot) + tran

    def get_vector(self):
        """Return coordinates as Vector.

        @return: coordinates as 3D vector
        @rtype: Vector
        """
        x, y, z = self.coord
        return Vector(x, y, z)

    def copy(self):
        """Create a copy of the Atom.

        Parent information is lost.
        """
        # Do a shallow copy then explicitly copy what needs to be deeper.
        shallow = copy.copy(self)
        shallow.parent = None
        shallow.coord = copy.copy(self.coord)
        shallow.xtra = self.xtra.copy()
        return shallow


class DisorderedAtom(DisorderedEntityWrapper):
    """Contains all Atom objects that represent the same disordered atom.

    One of these atoms is "selected" and all method calls not caught
    by DisorderedAtom are forwarded to the selected Atom object. In that way, a
    DisorderedAtom behaves exactly like a normal Atom. By default, the selected
    Atom object represents the Atom object with the highest occupancy, but a
    different Atom object can be selected by using the disordered_select(altloc)
    method.
    """

    def __init__(self, id):
        """Create DisorderedAtom.

        Arguments:
         - id - string, atom name
        """
        # TODO - make this a private attribute?
        self.last_occupancy = -999999
        super().__init__(id)

    # Special methods

    def __repr__(self):
        return "<Disordered Atom %s>" % self.id

    def disordered_add(self, atom):
        """Add a disordered atom."""
        # Add atom to dict, use altloc as key
        atom.disordered = 1
        # set the residue parent of the added atom
        atom.parent = self.parent
        self[atom.altloc] = atom
        if atom.occupancy > self.last_occupancy:
            self.last_occupancy = atom.occupancy
            self.disordered_select(atom.altloc)
