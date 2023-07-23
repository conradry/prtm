# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
Base class for Residue, Chain, Model and Structure classes.

It is a simple container class, with list and dictionary like properties.
"""
import logging
from abc import abstractmethod
from collections import OrderedDict
from copy import copy

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
