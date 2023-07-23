# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""Parser for PDB files."""
import logging
import re
from typing import Dict, NamedTuple, Tuple, Union

import numpy as np
from Bio import File
from Bio.File import as_handle

from .parser import Parser
from .structure_builder import StructureBuilder

logger = logging.getLogger(__name__)


class AtomData(NamedTuple):
    record_type: str
    fullname: str
    altloc: str
    resname: str
    chainid: str
    serial_number: int
    resseq: int
    icode: str
    hetero_flag: str
    coord: np.ndarray
    occupancy: float
    bfactor: float
    segid: str
    element: str
    name: str
    residue_id: Tuple[str, int, str]


# If PDB spec says "COLUMNS 18-20" this means line[17:20]


class PDBParser(Parser):
    """Parse a PDB file and return a Structure object."""

    # Private
    _error_message_counter: Dict[str, int]

    def __init__(self, PERMISSIVE=True, get_header=False, structure_builder=None):
        """Create a PDBParser object.

        The PDB parser call a number of standard methods in an aggregated
        StructureBuilder object. Normally this object is instanciated by the
        PDBParser object itself, but if the user provides his/her own
        StructureBuilder object, the latter is used instead.

        Arguments:
         - PERMISSIVE - Evaluated as a Boolean. If false, exceptions in
           constructing the SMCRA data structure are fatal. If true (DEFAULT),
           the exceptions are caught, but some residues or atoms will be missing.
           THESE EXCEPTIONS ARE DUE TO PROBLEMS IN THE PDB FILE!.
         - structure_builder - an optional user implemented StructureBuilder class.
        """
        if structure_builder is not None:
            self.structure_builder = structure_builder
        else:
            self.structure_builder = StructureBuilder()
        self.header = None
        self.trailer = None
        self.line_counter = 0
        self.PERMISSIVE = bool(PERMISSIVE)

        self.header = None
        self.trailer = None
        self._error_message_counter = {}

    # Public methods

    def get_structure(self, filename, structure_id: str = None):
        """Return the structure.

        Arguments:
         - id - string, the id that will be used for the structure
         - file - name of the PDB file OR an open filehandle
        """
        with as_handle(filename, mode="r") as handle:
            data = handle.readlines()

        self.header, coords_trailer = self._get_header(data)
        if structure_id is None:
            structure_id = self.header["id"]

        self.structure_builder.init_structure(structure_id)
        self.trailer = self._parse_coordinates(coords_trailer)
        self.structure_builder.set_header(self.header)
        structure = self.structure_builder.get_structure()

        return structure

    # Private methods

    def _get_header(self, header_coords_trailer):
        """Get the header of the PDB file, return the rest (PRIVATE)."""
        structure_builder = self.structure_builder
        i = 0
        for i in range(0, len(header_coords_trailer)):
            structure_builder.set_line_counter(i + 1)
            line = header_coords_trailer[i]
            record_type = line[0:6]
            if (
                record_type == "ATOM  "
                or record_type == "HETATM"
                or record_type == "MODEL "
            ):
                break
        header = header_coords_trailer[0:i]
        # Return the rest of the coords+trailer for further processing
        self.line_counter = i
        coords_trailer = header_coords_trailer[i:]
        header_dict = _parse_pdb_header_list(header)
        return header_dict, coords_trailer

    def _parse_coordinates(self, coords_trailer):
        """Parse the atomic data in the PDB file."""
        local_line_counter = 0
        structure_builder = self.structure_builder
        current_model_id = 0
        # Flag we have an open model
        model_open = 0
        current_chain_id = None
        current_segid = None
        current_residue_id = None
        current_resname = None
        for i in range(0, len(coords_trailer)):
            line = coords_trailer[i].rstrip("\n")
            record_type = line[0:6]
            global_line_counter = self.line_counter + local_line_counter + 1
            structure_builder.set_line_counter(global_line_counter)
            if record_type == "ATOM  " or record_type == "HETATM":
                # Initialize the Model - there was no explicit MODEL record
                if not model_open:
                    logger.debug("Adding new model: %s", current_model_id)
                    structure_builder.init_model(current_model_id)
                    current_model_id += 1
                    model_open = 1
                atom_data = self._parse_atom_line(line, global_line_counter)
                if current_segid != atom_data.segid:
                    current_segid = atom_data.segid
                    structure_builder.init_seg(current_segid)
                if current_chain_id != atom_data.chainid:
                    current_chain_id = atom_data.chainid
                    structure_builder.init_chain(current_chain_id)
                    current_residue_id = atom_data.residue_id
                    current_resname = atom_data.resname
                    try:
                        structure_builder.init_residue(
                            atom_data.resname,
                            atom_data.hetero_flag,
                            atom_data.resseq,
                            atom_data.icode,
                        )
                    except Exception as message:
                        self._handle_PDB_exception(message, global_line_counter)
                elif (
                    current_residue_id != atom_data.residue_id
                    or current_resname != atom_data.resname
                ):
                    current_residue_id = atom_data.residue_id
                    current_resname = atom_data.resname
                    try:
                        structure_builder.init_residue(
                            atom_data.resname,
                            atom_data.hetero_flag,
                            atom_data.resseq,
                            atom_data.icode,
                        )
                    except Exception as message:
                        self._handle_PDB_exception(message, global_line_counter)
                # init atom
                try:
                    structure_builder.init_atom(
                        atom_data.name,
                        atom_data.coord,
                        atom_data.bfactor,
                        atom_data.occupancy,
                        atom_data.altloc,
                        atom_data.fullname,
                        atom_data.serial_number,
                        atom_data.element,
                    )
                except Exception as message:
                    self._handle_PDB_exception(message, global_line_counter)
            elif record_type == "ANISOU":
                anisou = [
                    float(x)
                    for x in (
                        line[28:35],
                        line[35:42],
                        line[43:49],
                        line[49:56],
                        line[56:63],
                        line[63:70],
                    )
                ]
                # U's are scaled by 10^4
                anisou_array = np.array(anisou, np.float64) / 10000.0
                structure_builder.atom.anisou_array = anisou_array
            elif record_type == "MODEL ":
                try:
                    serial_num = int(line[10:14])
                except Exception:
                    self._handle_PDB_exception(
                        "Invalid or missing model serial number", global_line_counter
                    )
                    serial_num = 0
                structure_builder.init_model(current_model_id, serial_num)
                current_model_id += 1
                model_open = 1
                current_chain_id = None
                current_residue_id = None
            elif record_type == "END   " or record_type == "CONECT":
                # End of atomic data, return the trailer
                self.line_counter += local_line_counter
                return coords_trailer[local_line_counter:]
            elif record_type == "ENDMDL":
                model_open = 0
                current_chain_id = None
                current_residue_id = None
            elif record_type == "SIGUIJ":
                # standard deviation of anisotropic B factor
                siguij = [
                    float(x)
                    for x in (
                        line[28:35],
                        line[35:42],
                        line[42:49],
                        line[49:56],
                        line[56:63],
                        line[63:70],
                    )
                ]
                # U sigma's are scaled by 10^4
                siguij_array = np.array(siguij, np.float64) / 10000.0
                structure_builder.atom.set_siguij = siguij_array
            elif record_type == "SIGATM":
                # standard deviation of atomic positions
                sigatm = [
                    float(x)
                    for x in (
                        line[30:38],
                        line[38:45],
                        line[46:54],
                        line[54:60],
                        line[60:66],
                    )
                ]
                sigatm_array = np.array(sigatm, np.float64)
                structure_builder.atom.sigatm_array = sigatm_array
            local_line_counter += 1
        # EOF (does not end in END or CONECT)
        self.line_counter = self.line_counter + local_line_counter
        return []

    def _parse_atom_line(self, line: str, global_line_counter: int) -> AtomData:
        record_type = line[0:6]
        fullname = line[12:16]
        # get rid of whitespace in atom names
        split_list = fullname.split()
        if len(split_list) != 1:
            # atom name has internal spaces, e.g. " N B ", so
            # we do not strip spaces
            name = fullname
        else:
            # atom name is like " CA ", so we can strip spaces
            name = split_list[0]
        altloc = line[16]
        resname = line[17:20].strip(" ")
        chainid = line[20:22].strip(" ") or " "
        try:
            serial_number = int(line[6:11])
        except Exception:
            serial_number = 0
        resseq = int(line[22:26].split()[0])  # sequence identifier
        icode = line[26]  # insertion code
        if record_type == "HETATM":  # hetero atom flag
            if resname == "HOH" or resname == "WAT":
                hetero_flag = "W"
            else:
                hetero_flag = "H"
        else:
            hetero_flag = " "
        residue_id = (hetero_flag, resseq, icode)
        # atomic coordinates
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except Exception:
            # Should we allow parsing to continue in permissive mode?
            # If so, what coordinates should we default to?  Easier to abort!
            raise Exception(
                "Invalid or missing coordinate(s) at line %i." % global_line_counter
            )
        coord = np.array((x, y, z), np.float64)
        # occupancy & B factor
        try:
            occupancy = float(line[54:60])
        except Exception:
            self._handle_PDB_exception(
                "Invalid or missing occupancy", global_line_counter
            )
            occupancy = 0
        if occupancy is not None and occupancy < 0:
            # TODO - Should this be an error in strict mode?
            # self._handle_PDB_exception("Negative occupancy",
            #                            global_line_counter)
            # This uses fixed text so the warning occurs once only:
            logger.info("Negative occupancy in one or more atoms")
        try:
            bfactor = float(line[60:66])
        except Exception:
            self._handle_PDB_exception(
                "Invalid or missing B factor", global_line_counter
            )
            bfactor = 0.0  # The PDB use a default of zero if the data is missing
        segid = line[72:76]
        element = line[76:78].strip().upper()
        return AtomData(
            record_type,
            fullname,
            altloc,
            resname,
            chainid,
            serial_number,
            resseq,
            icode,
            hetero_flag,
            coord,
            occupancy,
            bfactor,
            segid,
            element,
            name,
            residue_id,
        )

    def _handle_PDB_exception(self, message, line_counter):
        """Handle exception (PRIVATE).

        This method catches an exception that occurs in the StructureBuilder
        object (if PERMISSIVE), or raises it again, this time adding the
        PDB line number to the error message.
        """
        count = self._error_message_counter.get(message, 0)
        message_full = "%s at line %i (occurence number %i)" % (
            message,
            line_counter,
            count,
        )
        if self.PERMISSIVE:
            if count <= 5:
                logger.warning("Exception: '%s'.", message_full)
            if count == 5:
                logger.warning("Future '%s' warnings will be ignored!", message)
            self._error_message_counter[message] = count + 1
        else:
            raise Exception(message_full)


def _get_journal(inl):
    # JRNL        AUTH   L.CHEN,M.DOI,F.S.MATHEWS,A.Y.CHISTOSERDOV,           2BBK   7
    journal = ""
    for l in inl:
        if re.search(r"\AJRNL", l):
            journal += l[19:72].lower()
    journal = re.sub(r"\s\s+", " ", journal)
    return journal


def _get_references(inl):
    # REMARK   1 REFERENCE 1                                                  1CSE  11
    # REMARK   1  AUTH   W.BODE,E.PAPAMOKOS,D.MUSIL                           1CSE  12
    references = []
    actref = ""
    for l in inl:
        if re.search(r"\AREMARK   1", l):
            if re.search(r"\AREMARK   1 REFERENCE", l):
                if actref != "":
                    actref = re.sub(r"\s\s+", " ", actref)
                    if actref != " ":
                        references.append(actref)
                    actref = ""
            else:
                actref += l[19:72].lower()

    if actref != "":
        actref = re.sub(r"\s\s+", " ", actref)
        if actref != " ":
            references.append(actref)
    return references


# bring dates to format: 1909-01-08
def _format_date(pdb_date):
    """Converts dates from DD-Mon-YY to YYYY-MM-DD format."""
    date = ""
    year = int(pdb_date[7:])
    if year < 50:
        century = 2000
    else:
        century = 1900
    date = str(century + year) + "-"
    all_months = [
        "xxx",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month = str(all_months.index(pdb_date[3:6]))
    if len(month) == 1:
        month = "0" + month
    date = date + month + "-" + pdb_date[:2]
    return date


def _chop_end_codes(line):
    """Chops lines ending with  '     1CSA  14' and the like."""
    return re.sub(r"\s\s\s\s+[\w]{4}.\s+\d*\Z", "", line)


def _chop_end_misc(line):
    """Chops lines ending with  '     14-JUL-97  1CSA' and the like."""
    return re.sub(r"\s\s\s\s+.*\Z", "", line)


def _nice_case(line):
    """Makes A Lowercase String With Capitals."""
    line = line.lower()
    s = ""
    i = 0
    nextCap = 1
    while i < len(line):
        c = line[i]
        if c >= "a" and c <= "z" and nextCap:
            c = c.upper()
            nextCap = 0
        elif (
            c == " "
            or c == "."
            or c == ","
            or c == ";"
            or c == ":"
            or c == "\t"
            or c == "-"
            or c == "_"
        ):
            nextCap = 1
        s += c
        i += 1
    return s


def parse_pdb_header(infile):
    """
    Returns the header lines of a pdb file as a dictionary.

    Dictionary keys are: head, deposition_date, release_date, structure_method,
    resolution, structure_reference, journal_reference, author and
    compound.
    """
    header = []
    with File.as_handle(infile, "r") as f:
        for l in f:
            record_type = l[0:6]
            if record_type in ("ATOM  ", "HETATM", "MODEL "):
                break
            else:
                header.append(l)
    return _parse_pdb_header_list(header)


def _parse_pdb_header_list(header):
    # database fields
    dict = {
        "id": header[0][62:66] if len(header) > 0 else "",
        "name": "",
        "head": "",
        "deposition_date": "1909-01-08",
        "release_date": "1909-01-08",
        "structure_method": "unknown",
        "resolution": 0.0,
        "structure_reference": "unknown",
        "journal_reference": "unknown",
        "author": "",
        "compound": {"1": {"misc": ""}},
        "source": {"1": {"misc": ""}},
    }

    dict["structure_reference"] = _get_references(header)
    dict["journal_reference"] = _get_journal(header)
    comp_molid = "1"
    # src_molid = "1"
    last_comp_key = "misc"
    last_src_key = "misc"
    remark_350_lines = []

    for hh in header:
        h = re.sub(r"[\s\n\r]*\Z", "", hh)  # chop linebreaks off
        # key=re.sub("\s.+\s*","",h)
        key = h[:6].strip()
        # tail=re.sub("\A\w+\s+\d*\s*","",h)
        tail = h[10:].strip()
        # print("%s:%s" % (key, tail)

        # From here, all the keys from the header are being parsed
        if key == "TITLE":
            name = _chop_end_codes(tail).lower()
            if "name" in dict:
                dict["name"] += " " + name
            else:
                dict["name"] = name
        elif key == "HEADER":
            rr = re.search(r"\d\d-\w\w\w-\d\d", tail)
            if rr is not None:
                dict["deposition_date"] = _format_date(_nice_case(rr.group()))
            head = _chop_end_misc(tail).lower()
            dict["head"] = head
        elif key == "COMPND":
            tt = re.sub(r"\;\s*\Z", "", _chop_end_codes(tail)).lower()
            # look for E.C. numbers in COMPND lines
            rec = re.search(r"\d+\.\d+\.\d+\.\d+", tt)
            if rec:
                dict["compound"][comp_molid]["ec_number"] = rec.group()
                tt = re.sub(r"\((e\.c\.)*\d+\.\d+\.\d+\.\d+\)", "", tt)
            tok = tt.split(":")
            if len(tok) >= 2:
                ckey = tok[0]
                cval = re.sub(r"\A\s*", "", tok[1])
                if ckey == "mol_id":
                    dict["compound"][cval] = {"misc": ""}
                    comp_molid = cval
                    last_comp_key = "misc"
                else:
                    dict["compound"][comp_molid][ckey] = cval
                    last_comp_key = ckey
            else:
                dict["compound"][comp_molid][last_comp_key] += tok[0] + " "
        elif key == "SOURCE":
            tt = re.sub(r"\;\s*\Z", "", _chop_end_codes(tail)).lower()
            tok = tt.split(":")
            # print(tok)
            if len(tok) >= 2:
                ckey = tok[0]
                cval = re.sub(r"\A\s*", "", tok[1])
                if ckey == "mol_id":
                    dict["source"][cval] = {"misc": ""}
                    comp_molid = cval
                    last_src_key = "misc"
                else:
                    dict["source"][comp_molid][ckey] = cval
                    last_src_key = ckey
            else:
                dict["source"][comp_molid][last_src_key] += tok[0] + " "
        elif key == "KEYWDS":
            kwd = _chop_end_codes(tail).lower()
            if "keywords" in dict:
                dict["keywords"] += " " + kwd
            else:
                dict["keywords"] = kwd
        elif key == "EXPDTA":
            expd = _chop_end_codes(tail)
            # chop junk at end of lines for some structures
            expd = re.sub(r"\s\s\s\s\s\s\s.*\Z", "", expd)
            # if re.search('\Anmr',expd,re.IGNORECASE): expd='nmr'
            # if re.search('x-ray diffraction',expd,re.IGNORECASE): expd='x-ray diffraction'
            dict["structure_method"] = expd.lower()
        elif key == "CAVEAT":
            # make Annotation entries out of these!!!
            pass
        elif key == "REVDAT":
            rr = re.search(r"\d\d-\w\w\w-\d\d", tail)
            if rr is not None:
                dict["release_date"] = _format_date(_nice_case(rr.group()))
        elif key == "JRNL":
            # print("%s:%s" % (key, tail))
            if "journal" in dict:
                dict["journal"] += tail
            else:
                dict["journal"] = tail
        elif key == "AUTHOR":
            auth = _nice_case(_chop_end_codes(tail))
            if "author" in dict:
                dict["author"] += auth
            else:
                dict["author"] = auth
        elif key == "REMARK":
            if re.search("REMARK   2 RESOLUTION.", hh):
                r = _chop_end_codes(re.sub("REMARK   2 RESOLUTION.", "", hh))
                r = re.sub(r"\s+ANGSTROM.*", "", r)
                try:
                    dict["resolution"] = float(r)
                except Exception:
                    # print('nonstandard resolution %r' % r)
                    dict["resolution"] = None
            elif hh.startswith("REMARK 350 "):
                remark_350_lines.append(hh)
        else:
            # print(key)
            pass
    if dict["structure_method"] == "unknown":
        if dict["resolution"] > 0.0:
            dict["structure_method"] = "x-ray diffraction"

    # if remark_350_lines:
    #    pr350 = ProcessRemark350()
    #    dict["bioassembly_data"] = pr350.process_lines(remark_350_lines)

    return dict
