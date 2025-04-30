from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from cgback.dataset import GraphDatasetItem


@dataclass
class GraphDataloaderItem:
    hu: Tensor
    xw: Tensor
    e: Tensor
    num_h: Tensor
    num_u: Tensor
    mask: Tensor


def collate_fn(batch: list[GraphDatasetItem]) -> GraphDataloaderItem:
    # Step 1: get data
    hu = torch.cat([data.hu for data in batch], dim=0)
    xw = torch.cat([data.xw for data in batch], dim=0)
    e = torch.cat([data.e for data in batch], dim=1)
    num_h = np.concatenate([data.num_h for data in batch])
    num_u = np.concatenate([data.num_u for data in batch])

    # Step 2: initialize mask
    mask = torch.zeros(xw.shape[0], 1).long()

    # Step 3: calculate offsets
    offset_node = 0
    offset_edge = 0
    for data in batch:
        num_data = data.num_h[0]
        num_node = data.num_h[0] + data.num_u[0]
        num_edge = data.e.shape[1]
        e[:, offset_edge: offset_edge + num_edge] += offset_node
        mask[offset_node: offset_node + num_data] = 1
        offset_node += num_node
        offset_edge += num_edge

    num_h = torch.from_numpy(num_h).to(torch.uint8)
    num_u = torch.from_numpy(num_u).to(torch.uint8)

    return GraphDataloaderItem(hu, xw, e, num_h, num_u, mask)
