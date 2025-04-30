import sys
import numpy as np
import dataclasses
import multiprocessing
from numpy.typing import NDArray
from scipy.spatial import KDTree
from cgback.parser import RESIDUE_TYPE_DECODER, HEAVY_ATOM_TYPE_ENCODER, HYDROGEN_ATOM_TYPE_ENCODER, BACKBONE_ATOM_TYPE_ENCODER
from cgback.system import System

RESIDUE_RING_GROUP = {
    "HIS": [np.array(sorted([HEAVY_ATOM_TYPE_ENCODER["HIS"]["CG"], HEAVY_ATOM_TYPE_ENCODER["HIS"]["ND1"], HEAVY_ATOM_TYPE_ENCODER["HIS"]["CE1"], HEAVY_ATOM_TYPE_ENCODER["HIS"]["NE2"], HEAVY_ATOM_TYPE_ENCODER["HIS"]["CD2"]]), dtype=np.uint64)],
    "PHE": [np.array(sorted([HEAVY_ATOM_TYPE_ENCODER["PHE"]["CG"], HEAVY_ATOM_TYPE_ENCODER["PHE"]["CD1"], HEAVY_ATOM_TYPE_ENCODER["PHE"]["CE1"], HEAVY_ATOM_TYPE_ENCODER["PHE"]["CD2"], HEAVY_ATOM_TYPE_ENCODER["PHE"]["CE2"], HEAVY_ATOM_TYPE_ENCODER["PHE"]["CZ"]]), dtype=np.uint64)],
    "PRO": [np.array(sorted([HEAVY_ATOM_TYPE_ENCODER["PRO"]["CA"], HEAVY_ATOM_TYPE_ENCODER["PRO"]["CB"], HEAVY_ATOM_TYPE_ENCODER["PRO"]["CG"], HEAVY_ATOM_TYPE_ENCODER["PRO"]["CD"], HEAVY_ATOM_TYPE_ENCODER["PRO"]["N"]]), dtype=np.uint64)],
    "TRP": [np.array(sorted([HEAVY_ATOM_TYPE_ENCODER["TRP"]["CG"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["CD1"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["NE1"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["CE2"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["CD2"]]), dtype=np.uint64),
            np.array(sorted([HEAVY_ATOM_TYPE_ENCODER["TRP"]["CD2"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["CE2"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["CE3"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["CZ2"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["CZ3"], HEAVY_ATOM_TYPE_ENCODER["TRP"]["CH2"]]), dtype=np.uint64)],
    "TYR": [np.array(sorted([HEAVY_ATOM_TYPE_ENCODER["TYR"]["CG"], HEAVY_ATOM_TYPE_ENCODER["TYR"]["CD1"], HEAVY_ATOM_TYPE_ENCODER["TYR"]["CE1"], HEAVY_ATOM_TYPE_ENCODER["TYR"]["CD2"], HEAVY_ATOM_TYPE_ENCODER["TYR"]["CE2"], HEAVY_ATOM_TYPE_ENCODER["TYR"]["CZ"]]), dtype=np.uint64)],
}

RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES = {
    "ALA": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["OXT", "C"],  # Heavy
    ],
    "ARG": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "NE"], ["NE", "CZ"], ["CZ", "NH1"], ["CZ", "NH2"], ["OXT", "C"],  # Heavy
    ],
    "ASN": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "OD1"], ["CG", "ND2"], ["OXT", "C"],  # Heavy
    ],
    "ASP": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "OD1"], ["CG", "OD2"], ["OXT", "C"],  # Heavy
    ],
    "CYS": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "SG"], ["OXT", "C"],  # Heavy
    ],
    "GLN": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "OE1"], ["CG", "NE2"], ["OXT", "C"],  # Heavy
    ],
    "GLU": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "OE1"], ["CG", "OE2"], ["OXT", "C"],  # Heavy
    ],
    "GLY": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["OXT", "C"],  # Heavy
    ],
    "HIS": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "ND1"], ["ND1", "CE1"], ["CE1", "NE2"], ["NE2", "CD2"], ["CD2", "CG"], ["OXT", "C"],  # Heavy
    ],
    "ILE": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG1"], ["CB", "CG2"], ["CG1", "CD1"], ["OXT", "C"],  # Heavy
    ],
    "LEU": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD1"], ["CG", "CD2"], ["OXT", "C"],  # Heavy
    ],
    "LYS": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "CE"], ["CE", "NZ"], ["OXT", "C"],  # Heavy
    ],
    "MET": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "SD"], ["SD", "CE"], ["OXT", "C"],
    ],
    "PHE": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD1"], ["CD1", "CE1"], ["CE1", "CZ"], ["CZ", "CE2"], ["CE2", "CD2"], ["CD2", "CG"], ["OXT", "C"],  # Heavy
    ],
    "PRO": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "N"], ["OXT", "C"],  # Heavy
    ],
    "SER": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "OG"], ["OXT", "C"],  # Heavy
    ],
    "THR": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "OG1"], ["CB", "CG2"], ["OXT", "C"],  # Heavy
    ],
    "TRP": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD2"], ["CD2", "CE3"], ["CE3", "CZ3"], ["CZ3", "CH2"], ["CH2", "CZ2"], ["CZ2", "CE2"], ["CE2", "CD2"], ["CE2", "NE1"], ["NE1", "CD1"], ["CD1", "CG"], ["OXT", "C"],  # Heavy
    ],
    "TYR": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG"], ["CG", "CD1"], ["CD1", "CE1"], ["CE1", "CZ"], ["CZ", "OH"], ["CZ", "CE2"], ["CE2", "CD2"], ["CD2", "CG"], ["OXT", "C"],
    ],
    "VAL": [
        ["N", "CA"], ["CA", "C"], ["C", "O"], ["CA", "CB"], ["CB", "CG1"], ["CB", "CG2"], ["OXT", "C"],  # Heavy
    ],
}
RESIDUE_HEAVY_HEAVY_BOND_ATOM_INDEXES = {
    "ALA": [[HEAVY_ATOM_TYPE_ENCODER["ALA"][atom1], HEAVY_ATOM_TYPE_ENCODER["ALA"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["ALA"]],
    "ARG": [[HEAVY_ATOM_TYPE_ENCODER["ARG"][atom1], HEAVY_ATOM_TYPE_ENCODER["ARG"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["ARG"]],
    "ASN": [[HEAVY_ATOM_TYPE_ENCODER["ASN"][atom1], HEAVY_ATOM_TYPE_ENCODER["ASN"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["ASN"]],
    "ASP": [[HEAVY_ATOM_TYPE_ENCODER["ASP"][atom1], HEAVY_ATOM_TYPE_ENCODER["ASP"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["ASP"]],
    "CYS": [[HEAVY_ATOM_TYPE_ENCODER["CYS"][atom1], HEAVY_ATOM_TYPE_ENCODER["CYS"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["CYS"]],
    "GLN": [[HEAVY_ATOM_TYPE_ENCODER["GLN"][atom1], HEAVY_ATOM_TYPE_ENCODER["GLN"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["GLN"]],
    "GLU": [[HEAVY_ATOM_TYPE_ENCODER["GLU"][atom1], HEAVY_ATOM_TYPE_ENCODER["GLU"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["GLU"]],
    "GLY": [[HEAVY_ATOM_TYPE_ENCODER["GLY"][atom1], HEAVY_ATOM_TYPE_ENCODER["GLY"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["GLY"]],
    "HIS": [[HEAVY_ATOM_TYPE_ENCODER["HIS"][atom1], HEAVY_ATOM_TYPE_ENCODER["HIS"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["HIS"]],
    "ILE": [[HEAVY_ATOM_TYPE_ENCODER["ILE"][atom1], HEAVY_ATOM_TYPE_ENCODER["ILE"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["ILE"]],
    "LEU": [[HEAVY_ATOM_TYPE_ENCODER["LEU"][atom1], HEAVY_ATOM_TYPE_ENCODER["LEU"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["LEU"]],
    "LYS": [[HEAVY_ATOM_TYPE_ENCODER["LYS"][atom1], HEAVY_ATOM_TYPE_ENCODER["LYS"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["LYS"]],
    "MET": [[HEAVY_ATOM_TYPE_ENCODER["MET"][atom1], HEAVY_ATOM_TYPE_ENCODER["MET"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["MET"]],
    "PHE": [[HEAVY_ATOM_TYPE_ENCODER["PHE"][atom1], HEAVY_ATOM_TYPE_ENCODER["PHE"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["PHE"]],
    "PRO": [[HEAVY_ATOM_TYPE_ENCODER["PRO"][atom1], HEAVY_ATOM_TYPE_ENCODER["PRO"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["PRO"]],
    "SER": [[HEAVY_ATOM_TYPE_ENCODER["SER"][atom1], HEAVY_ATOM_TYPE_ENCODER["SER"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["SER"]],
    "THR": [[HEAVY_ATOM_TYPE_ENCODER["THR"][atom1], HEAVY_ATOM_TYPE_ENCODER["THR"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["THR"]],
    "TRP": [[HEAVY_ATOM_TYPE_ENCODER["TRP"][atom1], HEAVY_ATOM_TYPE_ENCODER["TRP"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["TRP"]],
    "TYR": [[HEAVY_ATOM_TYPE_ENCODER["TYR"][atom1], HEAVY_ATOM_TYPE_ENCODER["TYR"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["TYR"]],
    "VAL": [[HEAVY_ATOM_TYPE_ENCODER["VAL"][atom1], HEAVY_ATOM_TYPE_ENCODER["VAL"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HEAVY_BOND_ATOM_NAMES["VAL"]],
}

RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES = {
    "ALA": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB1"], ["CB", "HB2"], ["CB", "HB3"],  # CB
    ],
    "ARG": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CG", "HG2"], ["CG", "HG3"],  # CG
        ["CD", "HD2"], ["CD", "HD3"],  # CD
        ["NE", "HE"],  # NE
        ["NH1", "HH11"], ["NH1", "HH12"],  # NH1
        ["NH2", "HH21"], ["NH2", "HH22"],  # NH2
    ],
    "ASN": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["ND2", "HD21"], ["ND2", "HD22"],  # ND2
    ],
    "ASP": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
    ],
    "CYS": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["SG", "HG"],  # SG
    ],
    "GLN": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CG", "HG2"], ["CG", "HG3"],  # CG
        ["NE2", "HE21"], ["NE2", "HE22"],  # NE2
    ],
    "GLU": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CG", "HG2"], ["CG", "HG3"],  # CG
    ],
    "GLY": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA2"], ["CA", "HA3"],  # CA
    ],
    "HIS": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["ND1", "HD1"],  # ND1
        ["CE1", "HE1"],  # CE1
        ["CD2", "HD2"],  # CD2
        ["NE2", "HE2"],  # NE2
    ],
    "ILE": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB"],  # CB
        ["CG1", "HG12"], ["CG1", "HG13"],  # CG1
        ["CD1", "HD11"], ["CD1", "HD12"], ["CD1", "HD13"],  # CD1
        ["CG2", "HG21"], ["CG2", "HG22"], ["CG2", "HG23"],  # CG2
    ],
    "LEU": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CG", "HG"],  # CG
        ["CD1", "HD11"], ["CD1", "HD12"], ["CD1", "HD13"],
        ["CD2", "HD21"], ["CD2", "HD22"], ["CD2", "HD23"],
    ],
    "LYS": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CG", "HG2"], ["CG", "HG3"],  # CG
        ["CD", "HD2"], ["CD", "HD3"],  # CD
        ["CE", "HE2"], ["CE", "HE3"],  # CE
        ["NZ", "HZ1"], ["NZ", "HZ2"], ["NZ", "HZ3"],  # NZ
    ],
    "MET": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CG", "HG2"], ["CG", "HG3"],  # CG
        ["CE", "HE1"], ["CE", "HE2"], ["CE", "HE3"],  # CE
    ],
    "PHE": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CD1", "HD1"],  # CD1
        ["CE1", "HE1"],  # CE1
        ["CD2", "HD2"],  # CD2
        ["CE2", "HE2"],  # CE2
        ["CZ", "HZ"],  # CZ
    ],
    "PRO": [
        ["N", "H1"], ["N", "H2"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CG", "HG2"], ["CG", "HG3"],  # CG
        ["CD", "HD2"], ["CD", "HD3"],  # CD
    ],
    "SER": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["OG", "HG"],  # OG
    ],
    "THR": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB"],  # CB
        ["OG1", "HG1"],  # OG1
        ["CG2", "HG21"], ["CG2", "HG22"], ["CG2", "HG23"],  # CG2
    ],
    "TRP": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CD1", "HD1"],  # CD1
        ["NE1", "HE1"],  # NE1
        ["CZ2", "HZ2"],  # CZ2
        ["CH2", "HH2"],  # CH2
        ["CZ3", "HZ3"],  # CZ3
        ["CE3", "HE3"],  # CE3
    ],
    "TYR": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB2"], ["CB", "HB3"],  # CB
        ["CD1", "HD1"],  # CD1
        ["CE1", "HE1"],  # CE1
        ["CD2", "HD2"],  # CD2
        ["CE2", "HE2"],  # CE2
        ["OH", "HH"],  # OH
    ],
    "VAL": [
        ["N", "H"], ["N", "H1"], ["N", "H2"], ["N", "H3"],  # N
        ["CA", "HA"],  # CA
        ["CB", "HB"],  # CB
        ["CG1", "HG11"], ["CG1", "HG12"], ["CG1", "HG13"],  # CG1
        ["CG2", "HG21"], ["CG2", "HG22"], ["CG2", "HG23"],  # CG2
    ],
}

RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_INDEXES = {
    "ALA": [[HEAVY_ATOM_TYPE_ENCODER["ALA"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["ALA"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["ALA"]],
    "ARG": [[HEAVY_ATOM_TYPE_ENCODER["ARG"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["ARG"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["ARG"]],
    "ASN": [[HEAVY_ATOM_TYPE_ENCODER["ASN"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["ASN"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["ASN"]],
    "ASP": [[HEAVY_ATOM_TYPE_ENCODER["ASP"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["ASP"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["ASP"]],
    "CYS": [[HEAVY_ATOM_TYPE_ENCODER["CYS"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["CYS"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["CYS"]],
    "GLN": [[HEAVY_ATOM_TYPE_ENCODER["GLN"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["GLN"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["GLN"]],
    "GLU": [[HEAVY_ATOM_TYPE_ENCODER["GLU"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["GLU"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["GLU"]],
    "GLY": [[HEAVY_ATOM_TYPE_ENCODER["GLY"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["GLY"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["GLY"]],
    "HIS": [[HEAVY_ATOM_TYPE_ENCODER["HIS"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["HIS"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["HIS"]],
    "ILE": [[HEAVY_ATOM_TYPE_ENCODER["ILE"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["ILE"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["ILE"]],
    "LEU": [[HEAVY_ATOM_TYPE_ENCODER["LEU"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["LEU"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["LEU"]],
    "LYS": [[HEAVY_ATOM_TYPE_ENCODER["LYS"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["LYS"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["LYS"]],
    "MET": [[HEAVY_ATOM_TYPE_ENCODER["MET"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["MET"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["MET"]],
    "PHE": [[HEAVY_ATOM_TYPE_ENCODER["PHE"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["PHE"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["PHE"]],
    "PRO": [[HEAVY_ATOM_TYPE_ENCODER["PRO"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["PRO"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["PRO"]],
    "SER": [[HEAVY_ATOM_TYPE_ENCODER["SER"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["SER"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["SER"]],
    "THR": [[HEAVY_ATOM_TYPE_ENCODER["THR"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["THR"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["THR"]],
    "TRP": [[HEAVY_ATOM_TYPE_ENCODER["TRP"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["TRP"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["TRP"]],
    "TYR": [[HEAVY_ATOM_TYPE_ENCODER["TYR"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["TYR"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["TYR"]],
    "VAL": [[HEAVY_ATOM_TYPE_ENCODER["VAL"][atom1], HYDROGEN_ATOM_TYPE_ENCODER["VAL"][atom2]] for atom1, atom2 in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_NAMES["VAL"]],
}


@dataclasses.dataclass
class RingDescriptor:
    ring_index: int
    group_index: int
    residue_index: int
    descriptor_index: int
    centroid: NDArray
    normal_vector: NDArray
    projections: NDArray


def calculate_centroid(ring_coordinates: NDArray) -> NDArray:
    centroid = np.mean(ring_coordinates, axis=0)

    return centroid


def calculate_normal_vector(ring_coordinates: NDArray, centroid: NDArray) -> NDArray:
    H = (ring_coordinates - centroid).T @ (ring_coordinates - centroid)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        print("[ERROR] Could not compute eigenvalues and eigenvectors. Structure probably has missing atoms.")
        sys.exit(0)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    return normal_vector


def bond_on_same_side(normal_vector: NDArray, centroid: NDArray, bond: NDArray) -> bool:
    distances = np.dot(bond - centroid, normal_vector)

    return np.prod(distances) >= 0


def calculate_intersection(normal_vector: NDArray, centroid: NDArray, bond: NDArray) -> NDArray:
    p0, p1 = np.array(bond)
    d = p1 - p0
    t = -np.dot(normal_vector, p0 - centroid) / np.dot(normal_vector, d)
    intersection = p0 + t * d

    return intersection


def calculate_projections(ring_coordinates: NDArray, normal_vector: NDArray, centroid: NDArray) -> NDArray:
    ring_vecs = ring_coordinates - centroid
    ring_dists = np.dot(ring_vecs, normal_vector)
    ring_dists = np.outer(ring_dists, normal_vector)
    projections = ring_coordinates - ring_dists

    return projections


def bond_inside_ring(intersection: NDArray, projections: NDArray) -> bool:
    angles = []
    for i in range(len(projections)):
        v1 = projections[i - 1] - intersection
        v2 = projections[i] - intersection
        angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
        angles.append(angle)
    total_angle = np.sum(angles)

    return np.isclose(total_angle, 2 * np.pi, rtol=0.01)


def penetration_detected(ring_descriptor: RingDescriptor, bond: NDArray) -> bool:
    penetration = False
    if not bond_on_same_side(ring_descriptor.normal_vector, ring_descriptor.centroid, bond):
        intersection = calculate_intersection(ring_descriptor.normal_vector, ring_descriptor.centroid, bond)
        if bond_inside_ring(intersection, ring_descriptor.projections):
            penetration = True

    return penetration


def find_residues_with_rings(system: System) -> list[int]:
    indexes = []
    for idx, residue_type in enumerate(system.residue_types):
        residue_name = RESIDUE_TYPE_DECODER[residue_type]
        if residue_name in ["HIS", "PHE", "PRO", "TRP", "TYR"]:
            indexes.append(idx)

    return indexes


def build_ring_neighbor_list(system: System, indexes: list[int], cutoff: float = 10.0) -> NDArray:
    ca_ring = system.heavy_coordinates[indexes, BACKBONE_ATOM_TYPE_ENCODER["CA"]]
    ca_all = system.heavy_coordinates[:, BACKBONE_ATOM_TYPE_ENCODER["CA"]]
    tree = KDTree(ca_all)
    max_neighbors = 0
    all_neighbors = []
    for idx, ca in zip(indexes, ca_ring):
        neighbors = tree.query_ball_point(ca, r=cutoff)
        neighbors = sorted(neighbors)
        neighbors = [jdx for jdx in neighbors if jdx != idx]
        all_neighbors.append(neighbors)
        max_neighbors = max(max_neighbors, len(neighbors))

    neighbor_list = np.full((len(indexes), max_neighbors), -1, dtype=np.int32)
    for idx, neighbors in enumerate(all_neighbors):
        neighbor_list[idx, :len(neighbors)] = neighbors

    return neighbor_list


def calculate_ring_descriptor(ring_coordinates: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    centroid = calculate_centroid(ring_coordinates)
    normal_vector = calculate_normal_vector(ring_coordinates, centroid)
    projections = calculate_projections(ring_coordinates, normal_vector, centroid)
    return centroid, normal_vector, projections


def initialize_ring_descriptors(system: System, ring_residue_indexes: list[int]) -> list[RingDescriptor]:
    ring_descriptors = []
    descriptor_index = 0
    for ring_index, residue_index in enumerate(ring_residue_indexes):
        residue_name = RESIDUE_TYPE_DECODER[system.residue_types[residue_index]]
        ring_groups = RESIDUE_RING_GROUP[residue_name]
        for group_index, ring_group in enumerate(ring_groups):
            centroid, normal_vector, projections = calculate_ring_descriptor(system.heavy_coordinates[residue_index, ring_group])
            ring_descriptor = RingDescriptor(ring_index, group_index, residue_index, descriptor_index, centroid, normal_vector, projections)
            ring_descriptors.append(ring_descriptor)
            descriptor_index += 1

    return ring_descriptors


def update_ring_descriptors(system: System, ring_descriptors: list[RingDescriptor], fix_mask: NDArray) -> None:
    for idx in range(len(ring_descriptors)):
        group_index = ring_descriptors[idx].group_index
        residue_index = ring_descriptors[idx].residue_index
        if fix_mask[residue_index]:
            residue_name = RESIDUE_TYPE_DECODER[system.residue_types[residue_index]]
            ring_group = RESIDUE_RING_GROUP[residue_name][group_index]
            centroid, normal_vector, projections = calculate_ring_descriptor(system.heavy_coordinates[residue_index, ring_group])
            ring_descriptors[idx].centroid = centroid
            ring_descriptors[idx].normal_vector = normal_vector
            ring_descriptors[idx].projections = projections


def find_penetration(system: System, ring_descriptor: RingDescriptor, neighbor_list: NDArray) -> tuple[int,int]:
    neighbors = neighbor_list[ring_descriptor.ring_index]
    for idx in neighbors:
        if idx == -1: break
        residue_name = RESIDUE_TYPE_DECODER[system.residue_types[idx]]

        for bond in RESIDUE_HEAVY_HEAVY_BOND_ATOM_INDEXES[residue_name]:
            bond = system.heavy_coordinates[idx, bond]
            if np.isnan(bond).any(): continue
            if penetration_detected(ring_descriptor, bond):
                return ring_descriptor.descriptor_index, int(idx)

        for bond in RESIDUE_HEAVY_HYDROGEN_BOND_ATOM_INDEXES[residue_name]:
            atom1 = system.heavy_coordinates[idx, bond[0]]
            atom2 = system.hydrogen_coordinates[idx, bond[1]]
            bond = np.array([atom1, atom2])
            if np.isnan(bond).any(): continue
            if penetration_detected(ring_descriptor, bond):
                return ring_descriptor.descriptor_index, int(idx)

    return ring_descriptor.descriptor_index, -1

def find_penetrations(system: System, ring_descriptors: list[RingDescriptor], neighbor_list: NDArray) -> list[tuple[int,int]]:
    with multiprocessing.Pool() as pool:
       results = pool.starmap(find_penetration, [(system, ring_descriptor, neighbor_list) for ring_descriptor in ring_descriptors])

    penetrations = [result for result in results if result[1] != -1]

    return penetrations
