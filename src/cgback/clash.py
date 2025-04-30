import numpy as np
import multiprocessing
from numpy.typing import NDArray
from scipy.spatial import KDTree
from cgback.parser import BACKBONE_ATOM_TYPE_ENCODER, RESIDUE_TYPE_ENCODER
from cgback.system import System

def build_clash_neighbor_list(system: System, cutoff: float = 20.0) -> NDArray:
    ca_all = system.heavy_coordinates[:, BACKBONE_ATOM_TYPE_ENCODER["CA"]]
    tree = KDTree(ca_all)
    pairs = tree.query_pairs(r=cutoff)
    pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
    neighbor_pairs = np.array(list(pairs), dtype=np.int32)

    return neighbor_pairs

def find_clash(system: System, i: int, j: int) -> tuple[int, int] | None:
    ri = system.heavy_coordinates[i, :, None, :]
    rj = system.heavy_coordinates[j, None, :, :]
    dr = np.linalg.norm(ri - rj, axis=-1)
    if abs(i - j) == 1:
        dr[:5, :] = np.nan
        dr[:, :5] = np.nan
    dr = dr.reshape(-1)
    if np.any(np.nan_to_num(dr, nan=np.inf) <= 1.2):
        return i, j

    hi = system.hydrogen_coordinates[i, :, None, :]
    hj = system.hydrogen_coordinates[j, None, :, :]
    dh = np.linalg.norm(hi - hj, axis=-1)
    if abs(i - j) == 1:
        if system.residue_types[i] == RESIDUE_TYPE_ENCODER["GLY"]:
            dh[:, :] = np.nan
        elif system.residue_types[i] == RESIDUE_TYPE_ENCODER["PRO"]:
            dh[:3, :] = np.nan
        else:
            dh[:, :5] = np.nan
        if system.residue_types[j] == RESIDUE_TYPE_ENCODER["GLY"]:
            dh[:, :] = np.nan
        elif system.residue_types[j] == RESIDUE_TYPE_ENCODER["PRO"]:
            dh[:, :3] = np.nan
        else:
            dh[:, :5] = np.nan
    dh = dh.reshape(-1)
    if np.any(np.nan_to_num(dh, nan=np.inf) <= 1.2):
        return i, j

    drh = np.linalg.norm(ri - hj, axis=-1)
    if abs(i - j) == 1:
        drh[:5, :] = np.nan
        if system.residue_types[j] == RESIDUE_TYPE_ENCODER["GLY"]:
            drh[:, :] = np.nan
        elif system.residue_types[j] == RESIDUE_TYPE_ENCODER["PRO"]:
            drh[:, :3] = np.nan
        else:
            drh[:, :5] = np.nan
    drh = drh.reshape(-1)
    if np.any(np.nan_to_num(drh, nan=np.inf) <= 1.2):
        return i, j

    dhr = np.linalg.norm(hi - rj, axis=-1)
    if abs(i - j) == 1:
        if system.residue_types[i] == RESIDUE_TYPE_ENCODER["GLY"]:
            dhr[:, :] = np.nan
        elif system.residue_types[i] == RESIDUE_TYPE_ENCODER["PRO"]:
            dhr[:3, :] = np.nan
        else:
            dhr[:, :5] = np.nan
        dhr[:, :5] = np.nan
    dhr = dhr.reshape(-1)
    if np.any(np.nan_to_num(dhr, nan=np.inf) <= 1.2):
        return i, j

    return None

def find_clashes(system: System, neighbor_list: NDArray) -> list[tuple[int,int]]:
    with multiprocessing.Pool() as pool:
        results = pool.starmap(find_clash, [(system, i, j) for i, j in neighbor_list])

    clashes = [result for result in results if result is not None]

    return clashes