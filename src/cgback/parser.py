import pathlib
import string
import itertools
import numpy as np
from logging import Logger
from multiprocessing import Pool
from pathlib import Path
from numpy.typing import NDArray
from collections import OrderedDict
from cgback.system import System

# Define residue types
RESIDUE_TYPE_ENCODER = OrderedDict([
    ("ALA", 0),
    ("ARG", 1),
    ("ASN", 2),
    ("ASP", 3),
    ("CYS", 4),
    ("GLN", 5),
    ("GLU", 6),
    ("GLY", 7),
    ("HIS", 8),
    ("ILE", 9),
    ("LEU", 10),
    ("LYS", 11),
    ("MET", 12),
    ("PHE", 13),
    ("PRO", 14),
    ("SER", 15),
    ("THR", 16),
    ("TRP", 17),
    ("TYR", 18),
    ("VAL", 19),
])
RESIDUE_TYPE_DECODER = [key for key in RESIDUE_TYPE_ENCODER.keys()]

# Define backbone types
BACKBONE_ATOM_TYPE_ENCODER = OrderedDict([
    ("N", 0),
    ("CA", 1),
    ("C", 2),
    ("O", 3),
    ("OXT", 4),
])
BACKBONE_ATOM_TYPE_DECODER = [key for key in BACKBONE_ATOM_TYPE_ENCODER.keys()]

# Define heavy atom types (in sync with backbone encoder/decoder)
HEAVY_ATOM_TYPE_ENCODER = OrderedDict([
    ("ALA", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5)])),
    ("ARG", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD", 7), ("NE", 8), ("CZ", 9), ("NH1", 10), ("NH2", 11)])),
    ("ASN", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("OD1", 7), ("ND2", 8)])),
    ("ASP", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("OD1", 7), ("OD2", 8)])),
    ("CYS", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("SG", 6)])),
    ("GLN", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD", 7), ("OE1", 8), ("NE2", 9)])),
    ("GLU", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD", 7), ("OE1", 8), ("OE2", 9)])),
    ("GLY", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4)])),
    ("HIS", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("ND1", 7), ("CE1", 8), ("NE2", 9), ("CD2", 10)])),
    ("ILE", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG1", 6), ("CD1", 7), ("CG2", 8)])),
    ("LEU", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD1", 7), ("CD2", 8)])),
    ("LYS", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD", 7), ("CE", 8), ("NZ", 9)])),
    ("MET", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("SD", 7), ("CE", 8)])),
    ("PHE", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD1", 7), ("CE1", 8), ("CZ", 9), ("CE2", 10), ("CD2", 11)])),
    ("PRO", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD", 7)])),
    ("SER", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("OG", 6)])),
    ("THR", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("OG1", 6), ("CG2", 7)])),
    ("TRP", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD1", 7), ("NE1", 8), ("CE2", 9), ("CZ2", 10), ("CH2", 11), ("CZ3", 12), ("CE3", 13), ("CD2", 14)])),
    ("TYR", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG", 6), ("CD1", 7), ("CE1", 8), ("CZ", 9), ("OH", 10), ("CE2", 11), ("CD2", 12)])),
    ("VAL", OrderedDict([("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("OXT", 4), ("CB", 5), ("CG1", 6), ("CG2", 7)])),
])
HEAVY_ATOM_TYPE_DECODER = [[key for key in val_dict.keys()] for val_dict in HEAVY_ATOM_TYPE_ENCODER.values()]

# Define hydrogen atom types
HYDROGEN_ATOM_TYPE_ENCODER = OrderedDict([
    ("ALA", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB1", 5), ("HB2", 6), ("HB3", 7)])),
    ("ARG", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HG2", 7), ("HG3", 8), ("HD2", 9), ("HD3", 10), ("HE", 11), ("HH11", 12), ("HH12", 13), ("HH21", 14), ("HH22", 15)])),
    ("ASN", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HD21", 7), ("HD22", 8)])),
    ("ASP", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6)])),
    ("CYS", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HG", 7)])),
    ("GLN", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HG2", 7), ("HG3", 8), ("HE21", 9), ("HE22", 10)])),
    ("GLU", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HG2", 7), ("HG3", 8)])),
    ("GLY", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA2", 4), ("HA3", 5)])),
    ("HIS", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HD1", 7), ("HE1", 8), ("HD2", 9), ("HE2", 10)])),
    ("ILE", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB", 5), ("HG12", 6), ("HG13", 7), ("HG21", 8), ("HG22", 9), ("HG23", 10), ("HD11", 11), ("HD12", 12), ("HD13", 13)])),
    ("LEU", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HG", 7), ("HD11", 8), ("HD12", 9), ("HD13", 10), ("HD21", 11), ("HD22", 12), ("HD23", 13)])),
    ("LYS", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HG2", 7), ("HG3", 8), ("HD2", 9), ("HD3", 10), ("HE2", 11), ("HE3", 12), ("HZ1", 13), ("HZ2", 14), ("HZ3", 15)])),
    ("MET", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HG2", 7), ("HG3", 8), ("HE1", 9), ("HE2", 10), ("HE3", 11)])),
    ("PHE", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HD1", 7), ("HD2", 8), ("HE1", 9), ("HE2", 10), ("HZ", 11)])),
    ("PRO", OrderedDict([("H1", 0), ("H2", 1), ("HA", 2), ("HB2", 3), ("HB3", 4), ("HG2", 5), ("HG3", 6), ("HD2", 7), ("HD3", 8)])),
    ("SER", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HG", 7)])),
    ("THR", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB", 5), ("HG1", 6), ("HG21", 7), ("HG22", 8), ("HG23", 9)])),
    ("TRP", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HD1", 7), ("HE1", 8), ("HE3", 9), ("HZ2", 10), ("HZ3", 11), ("HH2", 12)])),
    ("TYR", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB2", 5), ("HB3", 6), ("HD1", 7), ("HD2", 8), ("HE1", 9), ("HE2", 10), ("HH", 11)])),
    ("VAL", OrderedDict([("H1", 0), ("H2", 1), ("H3", 2), ("H", 3), ("HA", 4), ("HB", 5), ("HG11", 6), ("HG12", 7), ("HG13", 8), ("HG21", 9), ("HG22", 10), ("HG23", 11)])),
])
HYDROGEN_ATOM_TYPE_DECODER = [[key for key in val_dict.keys()] for val_dict in HYDROGEN_ATOM_TYPE_ENCODER.values()]

# Define constants to avoid enum overhead
LOC_TYPE_ENCODER = OrderedDict([
    ("NTER", 0b01),
    ("CTER", 0b10),
    ("MIDDLE", 0b00),
])
LOC_TYPE_DECODER = [key for key in LOC_TYPE_ENCODER.keys()]

# Define residue names with one letter
RESIDUE_TYPE_123 = OrderedDict([
    ("A", "ALA"),
    ("R", "ARG"),
    ("N", "ASN"),
    ("D", "ASP"),
    ("C", "CYS"),
    ("Q", "GLN"),
    ("E", "GLU"),
    ("G", "GLY"),
    ("H", "HIS"),
    ("I", "ILE"),
    ("L", "LEU"),
    ("K", "LYS"),
    ("M", "MET"),
    ("F", "PHE"),
    ("P", "PRO"),
    ("S", "SER"),
    ("T", "THR"),
    ("W", "TRP"),
    ("Y", "TYR"),
    ("V", "VAL"),
])
RESIDUE_TYPE_321 = OrderedDict([
    ("ALA", "A"),
    ("ARG", "R"),
    ("ASN", "N"),
    ("ASP", "D"),
    ("CYS", "C"),
    ("GLN", "Q"),
    ("GLU", "E"),
    ("GLY", "G"),
    ("HIS", "H"),
    ("ILE", "I"),
    ("LEU", "L"),
    ("LYS", "K"),
    ("MET", "M"),
    ("PHE", "F"),
    ("PRO", "P"),
    ("SER", "S"),
    ("THR", "T"),
    ("TRP", "W"),
    ("TYR", "Y"),
    ("VAL", "V"),
])

# Define alternative names in PDB files
ALTERNATIVE_RESIDUE_NAME = OrderedDict([
    ("HSD", "HIS"),
    ("HSE", "HIS"),
    ("HSP", "HIS"),
    ("HID", "HIS"),
    ("HIE", "HIS"),
    ("HIP", "HIS"),
])

ALTERNATIVE_HEAVY_ATOM_NAME = OrderedDict([
    ("ALA", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("ARG", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("ASN", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("ASP", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("CYS", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("GLN", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("GLU", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("GLY", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("HIS", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("ILE", OrderedDict([("OT1", "O"), ("OT2", "OXT"), ("CD", "CD1")])),
    ("LEU", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("LYS", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("MET", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("PHE", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("PRO", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("SER", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("THR", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("TRP", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("TYR", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
    ("VAL", OrderedDict([("OT1", "O"), ("OT2", "OXT")])),
])

ALTERNATIVE_HYDROGEN_ATOM_NAME = OrderedDict([
    ("ALA", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3")])),
    ("ARG", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3"), ("HG1", "HG3"), ("HD1", "HD3")])),
    ("ASN", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3")])),
    ("ASP", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3")])),
    ("CYS", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3")])),
    ("GLN", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3"), ("HG1", "HG3")])),
    ("GLU", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3"), ("HG1", "HG3")])),
    ("GLY", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HA1", "HA3")])),
    ("HIS", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3")])),
    ("ILE", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HG11", "HG13"), ("HD1", "HD11"), ("HD2", "HD12"), ("HD3", "HD13")])),
    ("LEU", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3")])),
    ("LYS", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3"), ("HG1", "HG3"), ("HD1", "HD3"), ("HE1", "HE3")])),
    ("MET", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3"), ("HG1", "HG3")])),
    ("PHE", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3")])),
    ("PRO", OrderedDict([("HT1", "H1"), ("HT2", "H2"), ("HB1", "HB3"), ("HG1", "HG3"), ("HD1", "HD3")])),
    ("SER", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3"), ("HG1", "HG")])),
    ("THR", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3")])),
    ("TRP", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3")])),
    ("TYR", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3"), ("HB1", "HB3")])),
    ("VAL", OrderedDict([("HN", "H"), ("HT1", "H1"), ("HT2", "H2"), ("HT3", "H3")])),
])


def pdb_atom_record_is_ca(pdb_atom_record: str) -> bool:
    return pdb_atom_record[12:16].strip() == "CA"


def cif_atom_record_is_ca(cif_atom_record: dict) -> bool:
    return cif_atom_record["_atom_site.label_atom_id"].strip() == "CA"


def count_ca_in_cif_atom_records(atom_records):
    with Pool() as pool:
        results = pool.starmap(cif_atom_record_is_ca, [(atom_record,) for atom_record in atom_records])
    return sum(results)


def count_ca_in_pdb_atom_records(atom_records):
    with Pool() as pool:
        results = pool.starmap(pdb_atom_record_is_ca, [(atom_record,) for atom_record in atom_records])
    return sum(results)


def determine_loc_types(chain_ids: NDArray) -> NDArray:
    num_residues = len(chain_ids)
    loc_types = np.full(num_residues, 255, dtype=np.uint8)
    for residue in range(num_residues):
        loc_type = LOC_TYPE_ENCODER["MIDDLE"]

        prev_idx = (residue - 1) % num_residues
        curr_idx = (residue + 0) % num_residues
        next_idx = (residue + 1) % num_residues

        chain_prev = chain_ids[prev_idx]
        chain_curr = chain_ids[curr_idx]
        chain_next = chain_ids[next_idx]

        # Case: multiple chains
        if chain_prev != chain_curr: loc_type |= LOC_TYPE_ENCODER["NTER"]
        if chain_curr != chain_next: loc_type |= LOC_TYPE_ENCODER["CTER"]
        # Case: single chain
        if chain_prev == chain_curr and prev_idx > curr_idx: loc_type |= LOC_TYPE_ENCODER["NTER"]
        if chain_curr == chain_next and curr_idx > next_idx: loc_type |= LOC_TYPE_ENCODER["CTER"]
        # Case: single chain made of a single residue
        if chain_prev == chain_curr and prev_idx == curr_idx: loc_type |= LOC_TYPE_ENCODER["NTER"] | LOC_TYPE_ENCODER["CTER"]

        loc_types[residue] = loc_type

    return loc_types


def determine_heavy_mask(residue_types: NDArray, loc_types: NDArray, heavy_coordinates: NDArray) -> NDArray:
    num_residues = len(residue_types)
    heavy_mask = np.full(num_residues, False, dtype=np.bool)
    for residue in range(num_residues):
        num_atoms = len(HEAVY_ATOM_TYPE_DECODER[residue_types[residue]])
        if loc_types[residue] & LOC_TYPE_ENCODER["CTER"] == LOC_TYPE_ENCODER["CTER"]:
            heavy_mask[residue] |= np.isnan(heavy_coordinates[residue, 0:num_atoms]).any()
        else:
            heavy_mask[residue] |= np.isnan(heavy_coordinates[residue, 0:4]).any()
            heavy_mask[residue] |= np.isnan(heavy_coordinates[residue, 5:num_atoms]).any()

    return heavy_mask


def determine_hydrogen_mask(residue_types: NDArray, loc_types: NDArray, hydrogen_coordinates: NDArray) -> NDArray:
    num_residues = len(residue_types)
    hydrogen_mask = np.full(num_residues, False, dtype=np.bool)
    for residue in range(num_residues):
        num_atoms = len(HYDROGEN_ATOM_TYPE_DECODER[residue_types[residue]])
        residue_name = RESIDUE_TYPE_DECODER[residue_types[residue]]
        if residue_name == "HIS":
            if loc_types[residue] & LOC_TYPE_ENCODER["NTER"] == LOC_TYPE_ENCODER["NTER"]:
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 0:3]).any()
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 4:7]).any()
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 8:10]).any()
                hydrogen_mask[residue] |= (np.isnan(hydrogen_coordinates[residue, 7:8]).any() & np.isnan(hydrogen_coordinates[residue, 10:num_atoms]).any())
            else:
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 3:7]).any()
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 8:10]).any()
                hydrogen_mask[residue] |= (np.isnan(hydrogen_coordinates[residue, 7:8]).any() & np.isnan(hydrogen_coordinates[residue, 10:num_atoms]).any())
        elif residue_name == "PRO":
            if loc_types[residue] & LOC_TYPE_ENCODER["NTER"] == LOC_TYPE_ENCODER["NTER"]:
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 0:num_atoms]).any()
            else:
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 2:num_atoms]).any()
        else:
            if loc_types[residue] & LOC_TYPE_ENCODER["NTER"] == LOC_TYPE_ENCODER["NTER"]:
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 0:3]).any()
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 4:num_atoms]).any()
            else:
                hydrogen_mask[residue] |= np.isnan(hydrogen_coordinates[residue, 3:num_atoms]).any()

    return hydrogen_mask


def system_from_pdb_path(pdb_path: Path, logger: Logger | None = None) -> System:
    # Read atom records
    with open(pdb_path, "r") as f:
        lines = f.readlines()
        lines = [line for line in lines if line.startswith("ATOM  ") or line.startswith("TER")]
        atom_records = [line for line in lines if line.startswith("ATOM  ")]

    # Count residues
    num_residues = count_ca_in_pdb_atom_records(atom_records)

    # Initialize arrays
    chain_ids = np.full(num_residues, 4294967295, dtype=np.uint32)
    residue_types = np.full(num_residues, 255, dtype=np.uint8)
    heavy_coordinates = np.full((num_residues, 15, 3), np.nan, dtype=np.float32)
    hydrogen_coordinates = np.full((num_residues, 16, 3), np.nan, dtype=np.float32)

    # Parse file
    current_chain_id = None
    chain = None
    current_residue_id = None
    residue = None
    for line_idx, line in enumerate(lines):
        if line.startswith("TER"):
            current_chain_id = None
            current_residue_id = None
            continue
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_id = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        try:
            new_residue_name = ALTERNATIVE_RESIDUE_NAME[residue_name]
            if new_residue_name != residue_name and logger is not None: logger.warning(f"Renaming {residue_name}:{residue} to {new_residue_name}:{residue}")
            residue_name = new_residue_name
        except KeyError:
            pass

        if current_residue_id != residue_id:
            current_residue_id = residue_id
            if residue is None:
                residue = 0
            else:
                residue += 1

        if current_chain_id != chain_id or lines[max(line_idx - 1, 0)].startswith("TER"):
            current_chain_id = chain_id
            if chain is None:
                chain = 0
            else:
                chain += 1

        chain_ids[residue] = chain
        residue_types[residue] = RESIDUE_TYPE_ENCODER[residue_name]

        if atom_name.startswith("H"):
            try:
                new_atom_name = ALTERNATIVE_HYDROGEN_ATOM_NAME[residue_name][atom_name]
                if new_atom_name != atom_name and logger is not None: logger.warning(f"Renaming {atom_name} to {new_atom_name} in {residue_name}:{residue}")
                atom_name = new_atom_name
            except KeyError:
                pass
            try:
                atom_type = HYDROGEN_ATOM_TYPE_ENCODER[residue_name][atom_name]
                hydrogen_coordinates[residue, atom_type] = np.array([x, y, z], dtype=np.float32)
            except KeyError:
                if logger is not None: logger.warning(f"Unknown atom name: {atom_name} in {residue_name}:{residue}")
        else:
            try:
                new_atom_name = ALTERNATIVE_HEAVY_ATOM_NAME[residue_name][atom_name]
                if new_atom_name != atom_name and logger is not None: logger.warning(f"Renaming {atom_name} to {new_atom_name} in {residue_name}:{residue}")
                atom_name = new_atom_name
            except KeyError:
                pass
            try:
                atom_type = HEAVY_ATOM_TYPE_ENCODER[residue_name][atom_name]
                heavy_coordinates[residue, atom_type] = np.array([x, y, z], dtype=np.float32)
            except KeyError:
                if logger is not None: logger.warning(f"Unknown atom name: {atom_name} in {residue_name}:{residue}")

    # Assign a location to each residue
    loc_types = determine_loc_types(chain_ids)

    # Assign masks
    heavy_mask = determine_heavy_mask(residue_types, loc_types, heavy_coordinates)
    hydrogen_mask = determine_hydrogen_mask(residue_types, loc_types, hydrogen_coordinates)

    return System(chain_ids, residue_types, loc_types, heavy_coordinates, hydrogen_coordinates, heavy_mask, hydrogen_mask)


def system_from_cif_path(cif_path: Path, logger: Logger | None = None) -> System:
    # Read atom records
    with open(cif_path, "r") as f:
        atom_data_started = False
        headers = []
        atom_records = []
        for line in f.readlines():
            if line.startswith("loop_"):
                atom_data_started = False
            elif line.startswith("_atom_site."):
                headers.append(line.strip())
                atom_data_started = True
            elif atom_data_started and not line.startswith("_"):
                values = line.strip().split()
                if len(values) == len(headers):
                    atom_record = dict(zip(headers, values))
                    atom_records.append(atom_record)

    # Count residues
    num_residues = count_ca_in_cif_atom_records(atom_records)

    # Initialize arrays
    chain_ids = np.full(num_residues, 4294967295, dtype=np.uint32)
    residue_types = np.full(num_residues, 255, dtype=np.uint8)
    heavy_coordinates = np.full((num_residues, 15, 3), np.nan, dtype=np.float32)
    hydrogen_coordinates = np.full((num_residues, 16, 3), np.nan, dtype=np.float32)

    # Parse file
    current_chain_id = None
    chain = None
    current_residue_id = None
    residue = None
    for atom_record in atom_records:
        atom_name = atom_record["_atom_site.label_atom_id"].strip()
        residue_name = atom_record["_atom_site.label_comp_id"].strip()
        chain_id = atom_record["_atom_site.label_asym_id"]
        residue_id = int(atom_record["_atom_site.label_seq_id"])
        x = float(atom_record["_atom_site.Cartn_x"])
        y = float(atom_record["_atom_site.Cartn_y"])
        z = float(atom_record["_atom_site.Cartn_z"])

        try:
            new_residue_name = ALTERNATIVE_RESIDUE_NAME[residue_name]
            if new_residue_name != residue_name and logger is not None: logger.warning(f"Renaming {residue_name}:{residue} to {new_residue_name}:{residue}")
            residue_name = new_residue_name
        except KeyError:
            pass

        if current_residue_id != residue_id:
            current_residue_id = residue_id
            if residue is None:
                residue = 0
            else:
                residue += 1

        if current_chain_id != chain_id:
            current_chain_id = chain_id
            if chain is None:
                chain = 0
            else:
                chain += 1

        chain_ids[residue] = chain
        residue_types[residue] = RESIDUE_TYPE_ENCODER[residue_name]

        if atom_name.startswith("H"):
            try:
                new_atom_name = ALTERNATIVE_HYDROGEN_ATOM_NAME[residue_name][atom_name]
                if new_atom_name != atom_name and logger is not None: logger.warning(f"Renaming {atom_name} to {new_atom_name} in {residue_name}:{residue}")
                atom_name = new_atom_name
            except KeyError:
                pass
            try:
                atom_type = HYDROGEN_ATOM_TYPE_ENCODER[residue_name][atom_name]
                hydrogen_coordinates[residue, atom_type] = np.array([x, y, z], dtype=np.float32)
            except KeyError:
                if logger is not None: logger.warning(f"Unknown atom name: {atom_name} in {residue_name}:{residue}")
        else:
            try:
                new_atom_name = ALTERNATIVE_HEAVY_ATOM_NAME[residue_name][atom_name]
                if new_atom_name != atom_name and logger is not None: logger.warning(f"Renaming {atom_name} to {new_atom_name} in {residue_name}:{residue}")
                atom_name = new_atom_name
            except KeyError:
                pass
            try:
                atom_type = HEAVY_ATOM_TYPE_ENCODER[residue_name][atom_name]
                heavy_coordinates[residue, atom_type] = np.array([x, y, z], dtype=np.float32)
            except KeyError:
                if logger is not None: logger.warning(f"Unknown atom name: {atom_name} in {residue_name}:{residue}")

    # Assign a location to each residue
    loc_types = determine_loc_types(chain_ids)

    # Assign masks
    heavy_mask = determine_heavy_mask(residue_types, loc_types, heavy_coordinates)
    hydrogen_mask = determine_hydrogen_mask(residue_types, loc_types, hydrogen_coordinates)

    return System(chain_ids, residue_types, loc_types, heavy_coordinates, hydrogen_coordinates, heavy_mask, hydrogen_mask)


def chain_id_generator():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    for length in itertools.count(1):
        for combo in itertools.product(chars, repeat=length):
            yield ''.join(combo)


def write_pdb_from_system(pdb_path: pathlib.Path, system: System) -> None:
    chain_names = chain_id_generator()
    with open(pdb_path, "w") as pdb:
        atom_id = 0
        residue_id = 0
        current_chain_id = None
        chain_name = next(chain_names)
        for heavy_coordinates, hydrogen_coordinates, residue_type, chain_id, loc_type in zip(system.heavy_coordinates, system.hydrogen_coordinates, system.residue_types, system.chain_ids, system.loc_types):
            if current_chain_id is None: current_chain_id = chain_id
            if current_chain_id != chain_id:
                current_chain_id = chain_id
                pdb.write("TER\n")
                residue_id = 0
                chain_name = next(chain_names)
            residue_name = RESIDUE_TYPE_DECODER[residue_type]
            for atom_type, atom_coordinates in enumerate(heavy_coordinates):
                if np.isnan(atom_coordinates).any(): continue
                x, y, z = atom_coordinates
                atom_name = HEAVY_ATOM_TYPE_DECODER[residue_type][atom_type]
                pdb.write(f"ATOM  {(atom_id + 1) % 100000:>5} {atom_name:<4} {residue_name:<3} {chain_name[0]}{(residue_id + 1) % 10000:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {atom_name[0]}  \n")
                atom_id += 1
            for atom_type, atom_coordinates in enumerate(hydrogen_coordinates):
                if np.isnan(atom_coordinates).any(): continue
                x, y, z = atom_coordinates
                atom_name = HYDROGEN_ATOM_TYPE_DECODER[residue_type][atom_type]
                pdb.write(f"ATOM  {(atom_id + 1) % 100000:>5} {atom_name:<4} {residue_name:<3} {chain_name[0]}{(residue_id + 1) % 10000:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {atom_name[0]}  \n")
                atom_id += 1
            residue_id += 1
        pdb.write("TER\n")
        pdb.write("END\n")


def write_cif_from_system(pdb_path: pathlib.Path, system: System) -> None:
    chain_names = chain_id_generator()
    with open(pdb_path, "w") as pdb:
        pdb.write(
            "data_\n"
            "loop_\n"
            "_atom_site.group_PDB\n"
            "_atom_site.id\n"
            "_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n"
            "_atom_site.label_alt_id\n"
            "_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n"
            "_atom_site.label_entity_id\n"
            "_atom_site.label_seq_id\n"
            "_atom_site.pdbx_PDB_ins_code\n"
            "_atom_site.Cartn_x\n"
            "_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n"
            "_atom_site.occupancy\n"
            "_atom_site.B_iso_or_equiv\n"
            "_atom_site.Cartn_x_esd\n"
            "_atom_site.Cartn_y_esd\n"
            "_atom_site.Cartn_z_esd\n"
            "_atom_site.occupancy_esd\n"
            "_atom_site.B_iso_or_equiv_esd\n"
            "_atom_site.pdbx_formal_charge\n"
            "_atom_site.auth_seq_id\n"
            "_atom_site.auth_comp_id\n"
            "_atom_site.auth_asym_id\n"
            "_atom_site.auth_atom_id\n"
            "_atom_site.pdbx_PDB_model_num\n"
        )
        atom_id = 0
        model_id = 0
        residue_id = 0
        current_chain_id = None
        chain_name = next(chain_names)
        for heavy_coordinates, hydrogen_coordinates, residue_type, chain_id, loc_type in zip(system.heavy_coordinates, system.hydrogen_coordinates, system.residue_types, system.chain_ids, system.loc_types):
            if current_chain_id is None: current_chain_id = chain_id
            if current_chain_id != chain_id:
                current_chain_id = chain_id
                residue_id = 0
                chain_name = next(chain_names)
            residue_name = RESIDUE_TYPE_DECODER[residue_type]
            for atom_type, atom_coordinates in enumerate(heavy_coordinates):
                if np.isnan(atom_coordinates).any(): continue
                x, y, z = atom_coordinates
                atom_name = HEAVY_ATOM_TYPE_DECODER[residue_type][atom_type]
                pdb.write(f"ATOM  {(atom_id + 1):>5} {atom_name[0]:<3} {atom_name:<4} . {residue_name:<4} {chain_name:1} ? {(residue_id + 1):>5} . {x:10.4f} {y:10.4f} {z:10.4f}  0.0  0.0  ?  ?  ?  ?  ?  .  {(residue_id + 1):>5} {residue_name:<4} {chain_name:1} {atom_name:>4} {(model_id + 1):>5}\n")
                atom_id += 1
            for atom_type, atom_coordinates in enumerate(hydrogen_coordinates):
                if np.isnan(atom_coordinates).any(): continue
                x, y, z = atom_coordinates
                atom_name = HYDROGEN_ATOM_TYPE_DECODER[residue_type][atom_type]
                pdb.write(f"ATOM  {(atom_id + 1):>5} {atom_name[0]:<3} {atom_name:<4} . {residue_name:<4} {chain_name:1} ? {(residue_id + 1):>5} . {x:10.4f} {y:10.4f} {z:10.4f}  0.0  0.0  ?  ?  ?  ?  ?  .  {(residue_id + 1):>5} {residue_name:<4} {chain_name:1} {atom_name:>4} {(model_id + 1):>5}\n")
                atom_id += 1
            residue_id += 1
