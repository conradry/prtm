import functools
import inspect
import logging
import os.path as op
import string
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Type
from urllib.parse import urlparse

from .parser import Parser
from .pdb_parser import PDBParser
from .structure import Structure


def get_rcsb_url(pdb_id: str, pdb_type: str) -> str:
    if pdb_type in ["pdb", "cif"]:
        url = f"http://files.rcsb.org/download/{pdb_id}.{pdb_type}.gz"
    elif pdb_type in ["mmtf"]:
        url = f"http://mmtf.rcsb.org/v1.0/full/{pdb_id}"
    else:
        raise TypeError(f"This route does not support '{pdb_type}' file format!")
    return url


def get_ebi_url(pdb_id: str, pdb_type: str) -> str:
    if pdb_type == "pdb":
        pdb_filename = f"pdb{pdb_id}.ent"
    elif pdb_type == "cif":
        pdb_filename = f"{pdb_id}.cif"
    else:
        raise TypeError(f"This route does not support '{pdb_type}' file format!")
    url = f"http://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_filename}"
    return url


def get_wwpdb_url(pdb_id: str, pdb_type: str) -> str:
    pdb_id_middle = pdb_id[1:3]
    if pdb_type == "pdb":
        pdb_filename = f"{pdb_id_middle}/pdb{pdb_id}.ent.gz"
        pdb_format = "pdb"
    elif pdb_type == "cif":
        pdb_filename = f"{pdb_id_middle}/{pdb_id}.cif.gz"
        pdb_format = "mmCIF"
    else:
        raise TypeError(f"This route does not support '{pdb_type}' file format!")
    url = f"ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/{pdb_format}/{pdb_filename}"
    return url


DEFAULT_ROUTES: Dict[str, Callable[[str, str], str]] = OrderedDict(
    [("rcsb", get_rcsb_url), ("ebi", get_ebi_url), ("wwpdb", get_wwpdb_url)]
)


def load(pdb_file: str, structure_id: str = None, **kwargs) -> Structure:
    """Load local PDB file.

    Args:
        pdb_file: File to load.
        kwargs: Optional keyword arguments to be passed to the parser
            ``__init__`` and ``get_structure`` methods.

    Load example:
        >>> import urllib.request
        >>> pdb_file = op.join(tempfile.gettempdir(), '4dkl.pdb')
        >>> r = urllib.request.urlretrieve('http://files.rcsb.org/download/4dkl.pdb', pdb_file)
        >>> load(pdb_file)
        <Structure id=4dkl>

    Fetch example:
        >>> load('wwpdb://4dkl')
        <Structure id=4dkl>
        >>> load('wwpdb://4dkl.cif')
        <Structure id=4dkl>
    """
    if isinstance(pdb_file, Path):
        pdb_file = pdb_file.as_posix()

    if structure_id is None:
        pdb_id = guess_pdb_id(pdb_file)
    else:
        pdb_id = structure_id
    pdb_type = guess_pdb_type(pdb_file)

    scheme = urlparse(pdb_file).scheme
    if scheme in DEFAULT_ROUTES:
        pdb_file = DEFAULT_ROUTES[scheme](pdb_id, pdb_type)

    parser = PDBParser(**kwargs)

    # with open_url(pdb_file) as fh:
    with open(pdb_file) as fh:
        structure = parser.get_structure(fh)
        if not structure.id:
            structure.id = pdb_id

    return structure


def guess_pdb_id(pdb_file: str) -> str:
    """Extract the PDB id from a PDB file.

    Examples
    --------
    >>> _guess_pdb_id('4dkl.pdb')
    '4dkl'
    >>> _guess_pdb_id('/data/structures/divided/pdb/26/pdb126d.ent.gz')
    '126d'
    >>> _guess_pdb_id('/tmp/100d.cif.gz')
    '100d'
    """
    pdb_id = op.basename(pdb_file)
    for extension in [".gz", ".pdb", ".ent", ".cif"]:
        pdb_id = pdb_id.partition(extension)[0]
    if len(pdb_id) == 7 and (pdb_id.startswith("ent") or pdb_id.startswith("pdb")):
        pdb_id = pdb_id[3:]
        assert len(pdb_id) == 4
    pdb_id = pdb_id.lower()
    pdb_id = pdb_id.replace(".", "")
    return pdb_id


def guess_pdb_type(pdb_file: str) -> str:
    """Guess PDB file type from file name.

    Examples
    --------
    >>> _guess_pdb_type('4dkl.pdb')
    'pdb'
    >>> _guess_pdb_type('/tmp/4dkl.cif.gz')
    'cif'
    """
    for suffix in reversed(Path(pdb_file).suffixes):
        suffix = suffix.lower().strip(string.digits)
        if suffix in [".pdb", ".ent"]:
            return "pdb"
        elif suffix in [".cif", ".mmcif"]:
            return "cif"
        elif suffix in [".mmtf"]:
            return "mmtf"
    raise Exception(f"Could not guess pdb type for file '{pdb_file}'!")


def get_parser(pdb_type: str, **kwargs) -> Parser:
    """Get `kmbio.PDB` parser appropriate for `pdb_type`."""
    MyParser: Type[Parser]
    if pdb_type == "pdb":
        MyParser = PDBParser
    elif pdb_type == "cif":
        kwargs.setdefault("use_auth_id", False)
        MyParser = MMCIFParser
    elif pdb_type == "mmtf":
        MyParser = MMTFParser
    else:
        raise Exception("Wrong pdb_type: '{}'".format(pdb_type))
    init_params = set(inspect.signature(MyParser).parameters)
    parser = MyParser(  # type: ignore
        **{k: kwargs.pop(k) for k in list(kwargs) if k in init_params}
    )
    func_params = set(inspect.signature(parser.get_structure).parameters)
    parser.get_structure = functools.partial(  # type: ignore
        parser.get_structure,
        **{k: kwargs.pop(k) for k in list(kwargs) if k in func_params},
    )
    if kwargs:
        warnings.warn(
            f"Not all arguments where used during the call to _get_parser! (kwargs = {kwargs})"
        )
    return parser
