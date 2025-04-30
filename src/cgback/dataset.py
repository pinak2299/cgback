from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch import Tensor
from numpy.typing import NDArray
from scipy.spatial import KDTree
from itertools import permutations
from cgback.parser import RESIDUE_TYPE_DECODER, RESIDUE_TYPE_321, System

ATOM_NAME_DICT = {
    "A": ["N", "CA", "C", "O", "OXT", "CB", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "R": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "PAD", "PAD", "PAD"],
    "N": ["N", "CA", "C", "O", "OXT", "CB", "CG", "OD1", "ND2", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "D": ["N", "CA", "C", "O", "OXT", "CB", "CG", "OD1", "OD2", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "C": ["N", "CA", "C", "O", "OXT", "CB", "SG", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "Q": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "OE1", "NE2", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "E": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "OE1", "OE2", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "G": ["N", "CA", "C", "O", "OXT", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "H": ["N", "CA", "C", "O", "OXT", "CB", "CG", "ND1", "CE1", "NE2", "CD2", "PAD", "PAD", "PAD", "PAD"],
    "I": ["N", "CA", "C", "O", "OXT", "CB", "CG1", "CD1", "CG2", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "L": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "CD2", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "K": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "CE", "NZ", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "M": ["N", "CA", "C", "O", "OXT", "CB", "CG", "SD", "CE", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "F": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2", "PAD", "PAD", "PAD"],
    "P": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "S": ["N", "CA", "C", "O", "OXT", "CB", "OG", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "T": ["N", "CA", "C", "O", "OXT", "CB", "OG1", "CG2", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
    "W": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
    "Y": ["N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2", "PAD", "PAD"],
    "V": ["N", "CA", "C", "O", "OXT", "CB", "CG1", "CG2", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"],
}

RESIDUE_ONE_TO_NUM_ATOMS = {
    "A": sum(atom != "PAD" for atom in ATOM_NAME_DICT["A"]),
    "R": sum(atom != "PAD" for atom in ATOM_NAME_DICT["R"]),
    "N": sum(atom != "PAD" for atom in ATOM_NAME_DICT["N"]),
    "D": sum(atom != "PAD" for atom in ATOM_NAME_DICT["D"]),
    "C": sum(atom != "PAD" for atom in ATOM_NAME_DICT["C"]),
    "Q": sum(atom != "PAD" for atom in ATOM_NAME_DICT["Q"]),
    "E": sum(atom != "PAD" for atom in ATOM_NAME_DICT["E"]),
    "G": sum(atom != "PAD" for atom in ATOM_NAME_DICT["G"]),
    "H": sum(atom != "PAD" for atom in ATOM_NAME_DICT["H"]),
    "I": sum(atom != "PAD" for atom in ATOM_NAME_DICT["I"]),
    "L": sum(atom != "PAD" for atom in ATOM_NAME_DICT["L"]),
    "K": sum(atom != "PAD" for atom in ATOM_NAME_DICT["K"]),
    "M": sum(atom != "PAD" for atom in ATOM_NAME_DICT["M"]),
    "F": sum(atom != "PAD" for atom in ATOM_NAME_DICT["F"]),
    "P": sum(atom != "PAD" for atom in ATOM_NAME_DICT["P"]),
    "S": sum(atom != "PAD" for atom in ATOM_NAME_DICT["S"]),
    "T": sum(atom != "PAD" for atom in ATOM_NAME_DICT["T"]),
    "W": sum(atom != "PAD" for atom in ATOM_NAME_DICT["W"]),
    "Y": sum(atom != "PAD" for atom in ATOM_NAME_DICT["Y"]),
    "V": sum(atom != "PAD" for atom in ATOM_NAME_DICT["V"]),
}


class Connectivity(Enum):
    N_TERMINAL = 0
    CENTRAL = 1
    C_TERMINAL = 2
    DISCONNECTED = 3


ATOM_NAME_LIST = sorted(list(set.union(*[set(names) for names in ATOM_NAME_DICT.values()])))
ATOM_NAME_LIST.remove("OXT")
ATOM_NAME_LIST.remove("PAD")
ATOM_ENCODER = {name: idx for idx, name in enumerate(ATOM_NAME_LIST)}
ATOM_DECODER = {idx: name for idx, name in enumerate(ATOM_NAME_LIST)}

RESIDUE_NAME_LIST = sorted(list(ATOM_NAME_DICT.keys()))
RESIDUE_ENCODER = {name: idx for idx, name in enumerate(RESIDUE_NAME_LIST)}
RESIDUE_DECODER = {idx: name for idx, name in enumerate(RESIDUE_NAME_LIST)}

CONNECTIVITY_ENCODER = {"n_terminal": 0, "central": 1, "c_terminal": 2, "disconnected": 3}
CONNECTIVITY_DECODER = {0: "n_terminal", 1: "central", 2: "c_terminal", 3: "disconnected"}


@dataclass
class GraphDatasetItem:
    hu: Tensor
    xw: Tensor
    e: Tensor
    num_h: NDArray
    num_u: NDArray


def process_mask(mask: str) -> NDArray:
    num_elem = len(mask)
    pad_mask = "+" + mask + "+"
    new_mask = np.full(num_elem, True)
    for idx in range(num_elem):
        if "-" in pad_mask[idx:idx + 3]:
            new_mask[idx] = False
    return new_mask


def build_dataset(coordinates: list[NDArray], sequences: list[str], masks: list[str], cutoff: float, chains: Optional[list[NDArray]] = None) -> dict[str, NDArray]:
    # Initialize chains
    if chains is None:
        chains = []
        for seq in sequences:
            chains.append(np.ones(len(seq), dtype=np.uint64))

    # Count the number of graphs
    total_h = 0
    total_u = 0
    total_g = 0
    neighbor_lists = []
    for crd, seq, msk in zip(coordinates, sequences, masks):
        msk = process_mask(msk)
        num_graphs = sum(msk)
        crd_ca = crd[:, 1, :]
        tree = KDTree(crd_ca)
        neighbor_list = []
        for ca in crd_ca:
            neighbors = tree.query_ball_point(ca, r=cutoff)
            neighbors = sorted(neighbors)
            neighbor_list.append(neighbors)
            total_u = total_u + len(neighbors)
        neighbor_lists.append(neighbor_list)
        # The total number is the sum of each graph minus CA and OXT
        total_h = total_h + sum(np.array([RESIDUE_ONE_TO_NUM_ATOMS[res] for res in seq]) * msk) - 2 * num_graphs
        total_g = total_g + num_graphs

    # Process data
    all_h = np.zeros((total_h, 3), dtype=np.uint8)
    all_u = np.zeros((total_u, 3), dtype=np.uint8)
    all_x = np.zeros((total_h, 3), dtype=np.float32)
    all_w = np.zeros((total_u, 3), dtype=np.float32)
    all_h_offset = np.zeros((total_g + 1,), dtype=np.uint64)
    all_u_offset = np.zeros((total_g + 1,), dtype=np.uint64)
    current_h = 0
    current_u = 0
    current_g = 0
    for crd, seq, ngh, chn, msk in zip(coordinates, sequences, neighbor_lists, chains, masks):
        if not msk: continue
        for idx, neighbors in enumerate(ngh):
            atm_f = []
            atm_x = []
            ctx_f = []
            ctx_x = []
            for jdx in neighbors:
                if idx == jdx:
                    for kdx, name in enumerate(ATOM_NAME_DICT[seq[jdx]]):
                        if name == "OXT": continue
                        if name == "PAD": continue
                        a = ATOM_ENCODER[name]
                        r = RESIDUE_ENCODER[seq[jdx]]
                        c = Connectivity.CENTRAL.value
                        f = np.array([a, r, c]).reshape((-1, 3))
                        x = crd[jdx][kdx].reshape((-1, 3))
                        if name == "CA":
                            ctx_f.insert(0, f)
                            ctx_x.insert(0, x)
                        else:
                            atm_f.append(f)
                            atm_x.append(x)
                else:
                    a = ATOM_ENCODER["CA"]
                    r = RESIDUE_ENCODER[seq[jdx]]
                    if idx == jdx + 1 and chn[idx] == chn[jdx]:
                        c = Connectivity.N_TERMINAL.value
                    elif idx == jdx - 1 and chn[idx] == chn[jdx]:
                        c = Connectivity.C_TERMINAL.value
                    else:
                        c = Connectivity.DISCONNECTED.value
                    f = np.array([a, r, c]).reshape((-1, 3))
                    x = crd[jdx][1].reshape((-1, 3))
                    ctx_f.append(f)
                    ctx_x.append(x)

            h = np.concatenate(atm_f)
            x = np.concatenate(atm_x)
            u = np.concatenate(ctx_f)
            w = np.concatenate(ctx_x)
            assert len(h) == len(x)
            assert len(u) == len(w)
            num_h = len(h)
            num_u = len(u)

            if np.isnan(x).any() or np.isnan(w).any():
                continue

            all_h[current_h: current_h + num_h] = h
            all_x[current_h: current_h + num_h] = x
            all_u[current_u: current_u + num_u] = u
            all_w[current_u: current_u + num_u] = w

            current_h = current_h + num_h
            current_u = current_u + num_u
            current_g = current_g + 1

            all_h_offset[current_g] = current_h
            all_u_offset[current_g] = current_u

    all_h = all_h[:current_h]
    all_u = all_u[:current_u]
    all_x = all_x[:current_h]
    all_w = all_w[:current_u]
    all_h_offset = all_h_offset[: current_g + 1]
    all_u_offset = all_u_offset[: current_g + 1]

    return {"h": all_h, "u": all_u, "x": all_x, "w": all_w, "h_offset": all_h_offset, "u_offset": all_u_offset}


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, system: System, cutoff: float) -> None:
        crd = np.random.randn(*system.heavy_coordinates.shape) + system.heavy_coordinates[:, 1:2, :]
        crd[:, 1, :] = system.heavy_coordinates[:, 1, :]
        seq = [residue_type for residue_type in system.residue_types]
        seq = [RESIDUE_TYPE_DECODER[res] for res in seq]
        seq = [RESIDUE_TYPE_321[res] for res in seq]
        seq = "".join(seq)
        msk = "+" * len(seq)
        chn = system.chain_ids

        self.data = build_dataset([crd], [seq], [msk], cutoff, chains=[chn])
        self.meta = {
            "atom_encoder": ATOM_ENCODER,
            "atom_decoder": ATOM_DECODER,
            "residue_encoder": RESIDUE_ENCODER,
            "residue_decoder": RESIDUE_DECODER,
            "connectivity_encoder": CONNECTIVITY_ENCODER,
            "connectivity_decoder": CONNECTIVITY_DECODER,
            "num_node_features": len(ATOM_ENCODER) + len(RESIDUE_ENCODER) + len(CONNECTIVITY_ENCODER),
            "num_graphs": len(self.data["h_offset"]) - 1,
        }

    def __len__(self):
        return self.meta["num_graphs"]

    def __getitem__(self, idx):
        h_offset = self.data["h_offset"]
        u_offset = self.data["u_offset"]

        h_lower = h_offset[idx]
        h_upper = h_offset[idx + 1]
        u_lower = u_offset[idx]
        u_upper = u_offset[idx + 1]

        h = torch.from_numpy(self.data["h"][h_lower:h_upper, :]).long()
        x = torch.from_numpy(self.data["x"][h_lower:h_upper, :]).float()
        u = torch.from_numpy(self.data["u"][u_lower:u_upper, :]).long()
        w = torch.from_numpy(self.data["w"][u_lower:u_upper, :]).float()
        e = torch.tensor(list(permutations(range(len(h) + len(u)), 2))).T

        c = w[0]
        x = x - c
        w = w - c

        h_a = torch.nn.functional.one_hot(h[:, 0], num_classes=len(self.meta["atom_encoder"]))
        h_r = torch.nn.functional.one_hot(h[:, 1], num_classes=len(self.meta["residue_encoder"]))
        h_c = torch.nn.functional.one_hot(h[:, 2], num_classes=len(self.meta["connectivity_encoder"]))
        h = torch.cat([h_a, h_r, h_c], dim=1)

        u_a = torch.nn.functional.one_hot(u[:, 0], num_classes=len(self.meta["atom_encoder"]))
        u_r = torch.nn.functional.one_hot(u[:, 1], num_classes=len(self.meta["residue_encoder"]))
        u_c = torch.nn.functional.one_hot(u[:, 2], num_classes=len(self.meta["connectivity_encoder"]))
        u = torch.cat([u_a, u_r, u_c], dim=1)

        hu = torch.cat([h, u], dim=0)
        xw = torch.cat([x, w], dim=0)
        num_h = np.array([len(h)])
        num_u = np.array([len(u)])

        return GraphDatasetItem(hu, xw, e, num_h, num_u)
