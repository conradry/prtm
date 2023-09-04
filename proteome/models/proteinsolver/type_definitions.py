import logging
import string
from abc import abstractmethod
from collections import OrderedDict
from copy import copy
from typing import List, NamedTuple

import numpy as np
import pandas as pd
from Bio.Data import IUPACData

_atom_name_dict = {}
_atom_name_dict["N"] = 1
_atom_name_dict["CA"] = 2
_atom_name_dict["C"] = 3
_atom_name_dict["O"] = 4

CHAIN_IDS = list(string.ascii_uppercase + string.digits + string.ascii_lowercase)
CHAIN_IDS += [(a + b) for a in CHAIN_IDS for b in CHAIN_IDS if a != b]

logger = logging.getLogger(__name__)


class Entity:
    """
    Basic container object. Structure, Model, Chain and Residue
    are subclasses of Entity. It deals with storage and lookup.
    """

    def __init__(self, id, children=None):
        self._id = id
        self._full_id = None
        self.parent = None
        self._children = OrderedDict()
        # Dictionary that keeps additional properties
        self.xtra = {}
        if children is not None:
            self.add(children)

    # Special methods

    def __getitem__(self, id):
        """Return the child with given id."""
        return self._children[id]

    def __setitem__(self, id, item):
        """Add a child."""
        assert id == item.id
        self.add([item])

    def __delitem__(self, id):
        """Remove a child."""
        child = self._children.pop(id)
        child.parent = None

    def __contains__(self, id):
        """True if there is a child element with the given id."""
        return id in self._children

    def __iter__(self):
        """Iterate over all children."""
        yield from self._children.values()

    def __len__(self):
        """Return the number of children."""
        return len(self._children)

    # Private methods

    def reset_full_id(self):
        """Reset the full_id.

        Sets the full_id of this entity and recursively of all its children to None.
        This means that it will be newly generated at the next call to get_full_id.
        """
        for child in self:
            try:
                child.reset_full_id()
            except AttributeError:
                pass  # Atoms do not cache their full ids
        self._full_id = None

    # Public methods

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        """Change the id of this entity.

        This will update the child_dict of this entity's parent
        and invalidate all cached full ids involving this entity.
        """
        if self.parent and new_id in self.parent:
            raise ValueError(
                "Cannot change id from `{0}` to `{1}`. "
                "The id `{1}` is already used for a sibling of this entity.".format(
                    self._id, new_id
                )
            )
        old_id = self._id
        self._id = new_id
        self.reset_full_id()
        if self.parent:
            pos = list(self.parent._children).index(old_id)
            # Need to copy self.parent because `del` sets it to None
            parent = self.parent
            del parent[old_id]
            self.parent = parent
            self.parent.insert(pos, self)

    @property
    @abstractmethod
    def level(self):
        """Return level in hierarchy.

        A - atom
        R - residue
        C - chain
        M - model
        S - structure
        """
        raise NotImplementedError

    def pop(self, id):
        """Remove and return a child."""
        child = self._children.pop(id)
        child.parent = None
        return child

    def clear(self):
        for child in self:
            child.parent = None
        self._children.clear()
        self.xtra.clear()

    def add(self, entities):
        """Add a child to the Entity."""
        # Single entity
        if entities is None:
            logger.info("Trying to add a 'None' child to {}".format(self))
            return
        if isinstance(entities, (Entity, DisorderedEntityWrapper)):
            entities = [entities]
        elif not isinstance(entities, list):
            # Like a generator...
            entities = list(entities)
        if any(entity.id in self for entity in entities):
            raise Exception("Some of the entities are defined twice")
        if len({entity.id for entity in entities}) < len(entities):
            raise Exception("Some of the entities are duplicates")
        for entity in entities:
            entity.parent = self
            self._children[entity.id] = entity

    def insert(self, pos, entities):
        """Add a child to the Entity at a specified position."""
        # Single entity
        if isinstance(entities, (Entity, DisorderedEntityWrapper)):
            entities = [entities]
        elif not isinstance(entities, list):
            # Like a generator...
            entities = list(entities)
        if any(c.id in self for c in entities):
            raise Exception("Some of the entities are defined twice")
        if len({c.id for c in entities}) < len(entities):
            raise Exception("Some of the entities are duplicates")
        self.add(entities)
        for id_ in list(self._children.keys())[pos : -len(entities)]:
            self._children.move_to_end(id_)

    @property
    def full_id(self):
        """Return the full id.

        The full id is a tuple containing all id's starting from
        the top object (Structure) down to the current object. A full id for
        a Residue object e.g. is something like:

        ("1abc", 0, "A", (" ", 10, "A"))

        This corresponds to:

        Structure with id "1abc"
        Model with id 0
        Chain with id "A"
        Residue with id (" ", 10, "A")

        The Residue id indicates that the residue is not a hetero-residue
        (or a water) because it has a blank hetero field, that its sequence
        identifier is 10 and its insertion code "A".
        """
        if self._full_id is None:
            entity_id = self.id
            lst = [entity_id]
            parent = self.parent
            while parent is not None:
                entity_id = parent.id
                lst.append(entity_id)
                parent = parent.parent
            lst.reverse()
            self._full_id = tuple(lst)
        return self._full_id

    def transform(self, rot, tran):
        """
        Apply rotation and translation to the atomic coordinates.

        Parameters
        ----------
        rot : `numpy.Array`
            A 3x3 rotation matrix.
        tran: `numpy.Array`
            A 1x3 translation vector.

        Examples
        --------
        >>> rotation=rotmat(pi, Vector(1, 0, 0))
        >>> translation=array((0, 0, 1))
        >>> entity.transform(rotation, translation)
        """
        for entity in self:
            entity.transform(rot, tran)

    def copy(self):
        shallow = copy(self)  # Copy class type, etc.
        # Need a generator from self because lazy evaluation:
        Entity.__init__(shallow, shallow.id, (c.copy() for c in self))
        shallow.xtra = self.xtra.copy()
        return shallow


class DisorderedEntityWrapper(object):
    """
    This class is a simple wrapper class that groups a number of equivalent
    Entities and forwards all method calls to one of them (the currently selected
    object). DisorderedResidue and DisorderedAtom are subclasses of this class.

    E.g.: A DisorderedAtom object contains a number of Atom objects,
    where each Atom object represents a specific position of a disordered
    atom in the structure.
    """

    def __init__(self, id):
        self.id = id
        self._siblings = {}
        self.selected_sibling = None
        self._parent = None
        self.disordered = 2

    # Special methods

    def __getattr__(self, method):
        """Forward the method call to the selected child."""
        if not hasattr(self, "selected_sibling"):
            # Avoid problems with pickling
            # Unpickling goes into infinite loop!
            raise AttributeError
        return getattr(self.selected_sibling, method)

    def __getitem__(self, id):
        """Return the child with the given id."""
        return self.selected_sibling[id]

    # XXX Why doesn't this forward to selected_sibling?
    # (NB: setitem was here before getitem, iter, len, sub)
    def __setitem__(self, id, child):
        """Add a child, associated with a certain id."""
        self._siblings[id] = child

    def __contains__(self, id):
        """True if the child has the given id."""
        return id in self.selected_sibling

    def __iter__(self):
        """Iterate over the children."""
        yield from self.selected_sibling

    def __len__(self):
        """Return the number of children."""
        return len(self.selected_sibling)

    def __sub__(self, other):
        """Subtraction with another object."""
        return self.selected_sibling - other

    # Public methods

    @property
    def parent(self):
        """Return parent."""
        return self._parent

    @parent.setter
    def parent(self, parent):
        if parent is None:
            # Detach parent
            self.parent = None
            for child in self.disordered_get_list():
                child.parent = None
        else:
            self._parent = parent

    @parent.setter
    def parent(self, parent):
        """Set the parent for the object and its children."""
        self._parent = parent
        for child in self.disordered_get_list():
            child.parent = parent

    def disordered_has_id(self, id):
        """True if there is an object present associated with this id."""
        return id in self._siblings

    def disordered_select(self, id):
        """Select the object with given id as the currently active object.

        Uncaught method calls are forwarded to the selected child object.
        """
        self.selected_sibling = self._siblings[id]

    def disordered_add(self, child):
        """This is implemented by DisorderedAtom and DisorderedResidue."""
        raise NotImplementedError

    def disordered_get_id_list(self):
        """Return a list of id's."""
        # Sort id list alphabetically
        return sorted(self._siblings)

    def disordered_get(self, id=None):
        """Get the child object associated with id.

        If id is None, the currently selected child is returned.
        """
        if id is None:
            return self.selected_sibling
        return self._siblings[id]

    def disordered_get_list(self):
        """Return list of children."""
        return list(self._siblings.values())


def m2rotaxis(m):
    """
    Return angles, axis pair that corresponds to rotation matrix m.
    """
    # Angle always between 0 and pi
    # Sense of rotation is defined by axis orientation
    t = 0.5 * (np.trace(m) - 1)
    t = max(-1, t)
    t = min(1, t)
    angle = np.arccos(t)
    if angle < 1e-15:
        # Angle is 0
        return 0.0, Vector(1, 0, 0)
    elif angle < np.pi:
        # Angle is smaller than pi
        x = m[2, 1] - m[1, 2]
        y = m[0, 2] - m[2, 0]
        z = m[1, 0] - m[0, 1]
        axis = Vector(x, y, z)
        axis.normalize()
        return angle, axis
    else:
        # Angle is pi - special case!
        m00 = m[0, 0]
        m11 = m[1, 1]
        m22 = m[2, 2]
        if m00 > m11 and m00 > m22:
            x = np.sqrt(m00 - m11 - m22 + 0.5)
            y = m[0, 1] / (2 * x)
            z = m[0, 2] / (2 * x)
        elif m11 > m00 and m11 > m22:
            y = np.sqrt(m11 - m00 - m22 + 0.5)
            x = m[0, 1] / (2 * y)
            z = m[1, 2] / (2 * y)
        else:
            z = np.sqrt(m22 - m00 - m11 + 0.5)
            x = m[0, 2] / (2 * z)
            y = m[1, 2] / (2 * z)
        axis = Vector(x, y, z)
        axis.normalize()
        return np.pi, axis


def vector_to_axis(line, point):
    """
    Returns the vector between a point and
    the closest point on a line (ie. the perpendicular
    projection of the point on the line).

    @type line: L{Vector}
    @param line: vector defining a line

    @type point: L{Vector}
    @param point: vector defining the point
    """
    line = line.normalized()
    norm = point.norm()
    angle = line.angle(point)
    return point - line ** (norm * np.cos(angle))


def rotaxis2m(theta, vector):
    """
    Calculate a left multiplying rotation matrix that rotates
    theta rad around vector.

    Example:

        >>> m=rotaxis(pi, Vector(1, 0, 0))
        >>> rotated_vector=any_vector.left_multiply(m)

    @type theta: float
    @param theta: the rotation angle


    @type vector: L{Vector}
    @param vector: the rotation axis

    @return: The rotation matrix, a 3x3 Numeric array.
    """
    vector = vector.copy()
    vector.normalize()
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    x, y, z = vector.get_array()
    rot = np.zeros((3, 3))
    # 1st row
    rot[0, 0] = t * x * x + c
    rot[0, 1] = t * x * y - s * z
    rot[0, 2] = t * x * z + s * y
    # 2nd row
    rot[1, 0] = t * x * y + s * z
    rot[1, 1] = t * y * y + c
    rot[1, 2] = t * y * z - s * x
    # 3rd row
    rot[2, 0] = t * x * z - s * y
    rot[2, 1] = t * y * z + s * x
    rot[2, 2] = t * z * z + c
    return rot


rotaxis = rotaxis2m


def refmat(p, q):
    """
    Return a (left multiplying) matrix that mirrors p onto q.

    Example:
        >>> mirror=refmat(p, q)
        >>> qq=p.left_multiply(mirror)
        >>> print(q)
        >>> print(qq) # q and qq should be the same

    @type p,q: L{Vector}
    @return: The mirror operation, a 3x3 Numeric array.
    """
    p.normalize()
    q.normalize()
    if (p - q).norm() < 1e-5:
        return np.identity(3)
    pq = p - q
    pq.normalize()
    b = pq.get_array()
    b.shape = (3, 1)
    i = np.identity(3)
    ref = i - 2 * np.dot(b, np.transpose(b))
    return ref


def rotmat(p, q):
    """
    Return a (left multiplying) matrix that rotates p onto q.

    Example:
        >>> r=rotmat(p, q)
        >>> print(q)
        >>> print(p.left_multiply(r))

    @param p: moving vector
    @type p: L{Vector}

    @param q: fixed vector
    @type q: L{Vector}

    @return: rotation matrix that rotates p onto q
    @rtype: 3x3 Numeric array
    """
    rot = np.dot(refmat(q, -p), refmat(p, -p))
    return rot


def calc_angle(v1, v2, v3):
    """
    Calculate the angle between 3 vectors
    representing 3 connected points.

    @param v1, v2, v3: the tree points that define the angle
    @type v1, v2, v3: L{Vector}

    @return: angle
    @rtype: float
    """
    v1 = v1 - v2
    v3 = v3 - v2
    return v1.angle(v3)


def calc_dihedral(v1, v2, v3, v4):
    """
    Calculate the dihedral angle between 4 vectors
    representing 4 connected points. The angle is in
    ]-pi, pi].

    @param v1, v2, v3, v4: the four points that define the dihedral angle
    @type v1, v2, v3, v4: L{Vector}
    """
    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3
    u = ab**cb
    v = db**cb
    w = u**v
    angle = u.angle(v)
    # Determine sign of angle
    try:
        if cb.angle(w) > 0.001:
            angle = -angle
    except ZeroDivisionError:
        # dihedral=pi
        pass
    return angle


class Vector(object):
    "3D vector"

    def __init__(self, x, y=None, z=None):
        if y is None and z is None:
            # Array, list, tuple...
            if len(x) != 3:
                raise ValueError("Vector: x is not a " "list/tuple/array of 3 numbers")
            self._ar = np.array(x, "d")
        else:
            # Three numbers
            self._ar = np.array((x, y, z), "d")

    def __repr__(self):
        x, y, z = self._ar
        return "<Vector %.2f, %.2f, %.2f>" % (x, y, z)

    def __neg__(self):
        "Return Vector(-x, -y, -z)"
        a = -self._ar
        return Vector(a)

    def __add__(self, other):
        "Return Vector+other Vector or scalar"
        if isinstance(other, Vector):
            a = self._ar + other._ar
        else:
            a = self._ar + np.array(other)
        return Vector(a)

    def __sub__(self, other):
        "Return Vector-other Vector or scalar"
        if isinstance(other, Vector):
            a = self._ar - other._ar
        else:
            a = self._ar - np.array(other)
        return Vector(a)

    def __mul__(self, other):
        "Return Vector.Vector (dot product)"
        return sum(self._ar * other._ar)

    def __div__(self, x):
        "Return Vector(coords/a)"
        a = self._ar / np.array(x)
        return Vector(a)

    def __pow__(self, other):
        "Return VectorxVector (cross product) or Vectorxscalar"
        if isinstance(other, Vector):
            a, b, c = self._ar
            d, e, f = other._ar
            c1 = np.linalg.det(np.array(((b, c), (e, f))))
            c2 = -np.linalg.det(np.array(((a, c), (d, f))))
            c3 = np.linalg.det(np.array(((a, b), (d, e))))
            return Vector(c1, c2, c3)
        else:
            a = self._ar * np.array(other)
            return Vector(a)

    def __getitem__(self, i):
        return self._ar[i]

    def __setitem__(self, i, value):
        self._ar[i] = value

    def __contains__(self, i):
        return i in self._ar

    def norm(self):
        "Return vector norm"
        return np.sqrt(sum(self._ar * self._ar))

    def normsq(self):
        "Return square of vector norm"
        return abs(sum(self._ar * self._ar))

    def normalize(self):
        "Normalize the Vector"
        self._ar = self._ar / self.norm()

    def normalized(self):
        "Return a normalized copy of the Vector"
        v = self.copy()
        v.normalize()
        return v

    def angle(self, other):
        "Return angle between two vectors"
        n1 = self.norm()
        n2 = other.norm()
        c = (self * other) / (n1 * n2)
        # Take care of roundoff errors
        c = min(c, 1)
        c = max(-1, c)
        return np.arccos(c)

    def get_array(self):
        "Return (a copy of) the array of coordinates"
        return np.array(self._ar)

    def left_multiply(self, matrix):
        "Return Vector=Matrix x Vector"
        a = np.dot(matrix, self._ar)
        return Vector(a)

    def right_multiply(self, matrix):
        "Return Vector=Vector x Matrix"
        a = np.dot(self._ar, matrix)
        return Vector(a)

    def copy(self):
        "Return a deep copy of the Vector"
        return Vector(self._ar)


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
        shallow = copy(self)
        shallow.parent = None
        shallow.coord = copy(self.coord)
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


class Model(Entity):
    """
    The object representing a model in a structure. In a structure
    derived from an X-ray crystallography experiment, only a single
    model will be present (with some exceptions). NMR structures
    normally contain many different models.
    """

    level = "M"

    def __init__(self, id, serial_num=None, **kwargs):
        """
        Arguments:
        o id - int
        o serial_num - int
        """
        if serial_num is None:
            self.serial_num = id
        else:
            self.serial_num = serial_num
        super().__init__(id, **kwargs)

    # Private methods

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __ge__(self, other):
        return self.id >= other.id

    def __gt__(self, other):
        return self.id > other.id

    # Special methods

    def __repr__(self):
        return "<Model id=%s>" % self.id

    # Public

    @property
    def chains(self):
        for c in self:
            yield c

    @property
    def residues(self):
        for c in self:
            for r in c:
                yield r

    @property
    def atoms(self):
        for r in self.residues:
            for a in r:
                yield a

    # Custom
    def extract(self, chain_ids):
        model = Model(self.id, self.serial_num)
        for chain_id in chain_ids:
            model.add(self[chain_id].copy())
        return model


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


class StructureBuilder(object):
    """
    Deals with contructing the Structure object. The StructureBuilder class is used
    by the PDBParser classes to translate a file to a Structure object.
    """

    def __init__(self):
        self.line_counter = 0
        self.header = {}

    def _is_completely_disordered(self, residue):
        "Return 1 if all atoms in the residue have a non blank altloc."
        atom_list = residue.get_unpacked_list()
        for atom in atom_list:
            altloc = atom.altloc
            if altloc == " ":
                return 0
        return 1

    # Public methods called by the Parser classes

    def set_header(self, header):
        self.header = header

    def set_line_counter(self, line_counter):
        """
        The line counter keeps track of the line in the PDB file that
        is being parsed.

        Arguments:
        o line_counter - int
        """
        self.line_counter = line_counter

    def init_structure(self, structure_id):
        """Initiate a new Structure object with given id.

        Arguments:
        o id - string
        """
        self.structure = Structure(structure_id)

    def init_model(self, model_id, serial_num=None):
        """Initiate a new Model object with given id.

        Arguments:
        o id - int
        o serial_num - int
        """
        self.model = Model(model_id, serial_num)
        self.structure.add(self.model)

    def init_chain(self, chain_id):
        """Initiate a new Chain object with given id.

        Arguments:
        o chain_id - string
        """
        if chain_id in self.model:
            self.chain = self.model[chain_id]
            logger.info(
                "WARNING: Chain %s is discontinuous at line %i.",
                chain_id,
                self.line_counter,
            )
        else:
            self.chain = Chain(chain_id)
            self.model.add(self.chain)

    def init_seg(self, segid):
        """Flag a change in segid.

        Arguments:
        o segid - string
        """
        self.segid = segid

    def init_residue(self, resname, field, resseq, icode):
        """
        Initiate a new Residue object.

        Arguments:

            - resname - string, e.g. "ASN"
            - field - hetero flag, "W" for waters, "H" for
              hetero residues, otherwise blank.
            - resseq - int, sequence identifier
            - icode - string, insertion code
        """
        if field != " ":
            if field == "H":
                # The hetero field consists of H_ + the residue name (e.g. H_FUC)
                field = "H_" + resname
        res_id = (field, resseq, icode)
        if field == " ":
            if res_id in self.chain:
                # There already is a residue with the id (field, resseq, icode).
                # This only makes sense in the case of a point mutation.
                logger.info(
                    "Residue ('%s', %i, '%s') redefined at line %i.",
                    field,
                    resseq,
                    icode,
                    self.line_counter,
                )
                duplicate_residue = self.chain[res_id]
                if isinstance(duplicate_residue, DisorderedResidue):
                    # The residue in the chain is a DisorderedResidue object.
                    # So just add the last Residue object.
                    if duplicate_residue.disordered_has_id(resname):
                        # The residue was already made
                        self.residue = duplicate_residue
                        duplicate_residue.disordered_select(resname)
                    else:
                        # Make a new residue and add it to the already
                        # present DisorderedResidue
                        new_residue = Residue(res_id, resname, self.segid)
                        duplicate_residue.disordered_add(new_residue)
                        self.residue = duplicate_residue
                        return
                else:
                    if resname == duplicate_residue.resname:
                        logger.warning(
                            "Residue '%s' ('%s', %i, '%s') is already defined in chain '%s' "
                            "with the same name at line %i.",
                            resname,
                            field,
                            resseq,
                            icode,
                            self.chain.id,
                            self.line_counter,
                        )
                        self.residue = duplicate_residue
                        return
                    # Make a new DisorderedResidue object and put all
                    # the Residue objects with the id (field, resseq, icode) in it.
                    # These residues each should have non-blank altlocs for all their atoms.
                    # If not, the PDB file probably contains an error.
                    if not self._is_completely_disordered(duplicate_residue):
                        # if this exception is ignored, a residue will be missing
                        self.residue = None
                        raise Exception(
                            "Blank altlocs in duplicate residue %s ('%s', %i, '%s')"
                            % (resname, field, resseq, icode)
                        )
                    del self.chain[res_id]
                    new_residue = Residue(res_id, resname, self.segid)
                    disordered_residue = DisorderedResidue(res_id)
                    self.chain.add(disordered_residue)
                    disordered_residue.disordered_add(duplicate_residue)
                    disordered_residue.disordered_add(new_residue)
                    self.residue = disordered_residue
                    return
        self.residue = Residue(res_id, resname, self.segid)
        self.chain.add(self.residue)

    def init_atom(
        self,
        name,
        coord,
        b_factor,
        occupancy,
        altloc,
        fullname,
        serial_number=None,
        element=None,
    ):
        """
        Initiate a new Atom object.

        Arguments:
        o name - string, atom name, e.g. CA, spaces should be stripped
        o coord - Numeric array (Float0, size 3), atomic coordinates
        o b_factor - float, B factor
        o occupancy - float
        o altloc - string, alternative location specifier
        o fullname - string, atom name including spaces, e.g. " CA "
        o element - string, upper case, e.g. "HG" for mercury
        """
        residue = self.residue
        # if residue is None, an exception was generated during
        # the construction of the residue
        if residue is None:
            return
        # First check if this atom is already present in the residue.
        # If it is, it might be due to the fact that the two atoms have atom
        # names that differ only in spaces (e.g. "CA.." and ".CA.",
        # where the dots are spaces). If that is so, use all spaces
        # in the atom name of the current atom.
        if name in residue:
            duplicate_atom = residue[name]
            # atom name with spaces of duplicate atom
            duplicate_fullname = duplicate_atom.fullname
            if duplicate_fullname != fullname:
                # name of current atom now includes spaces
                name = fullname
                logger.info(
                    "Atom names %r and %r differ only in spaces at line %i.",
                    duplicate_fullname,
                    fullname,
                    self.line_counter,
                )
        self.atom = Atom(
            name, coord, b_factor, occupancy, altloc, fullname, serial_number, element
        )
        if altloc != " ":
            # The atom is disordered
            if name in residue:
                # Residue already contains this atom
                duplicate_atom = residue[name]
                if isinstance(duplicate_atom, DisorderedAtom):
                    duplicate_atom.disordered_add(self.atom)
                else:
                    # This is an error in the PDB file:
                    # a disordered atom is found with a blank altloc
                    # Detach the duplicate atom, and put it in a
                    # DisorderedAtom object together with the current
                    # atom.
                    del residue[name]
                    disordered_atom = DisorderedAtom(name)
                    residue.add(disordered_atom)
                    disordered_atom.disordered_add(self.atom)
                    disordered_atom.disordered_add(duplicate_atom)
                    residue.disordered = 1
                    logger.info(
                        "WARNING: disordered atom found with blank altloc before line %i.",
                        self.line_counter,
                    )
            else:
                # The residue does not contain this disordered atom
                # so we create a new one.
                disordered_atom = DisorderedAtom(name)
                residue.add(disordered_atom)
                # Add the real atom to the disordered atom, and the
                # disordered atom to the residue
                disordered_atom.disordered_add(self.atom)
                # TODO: Setting `residue.disordered = ` without checking types causes
                # one of the tests to fail. But the whole disordered = {0, 1, 2} is stupid.
                if isinstance(residue, Residue):
                    residue.disordered = 1
        else:
            # The atom is not disordered
            residue.add(self.atom)

    def get_structure(self):
        "Return the structure."
        # first sort everything
        # self.structure.sort()
        # Add the header dict
        self.structure.header = self.header
        return self.structure

    def set_symmetry(self, spacegroup, cell):
        pass
