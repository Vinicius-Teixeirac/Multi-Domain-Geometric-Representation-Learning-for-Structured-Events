"""Serialization helpers for PyTorch Geometric graph objects."""

from pathlib import Path
from typing import Union

import torch
from torch_geometric.data import Data, HeteroData


def save_graph(graph: Union[Data, HeteroData], path: Path) -> None:
    """
    Persist a PyG Data/HeteroData graph to disk, creating parent dirs as needed.

    Parameters
    ----------
    graph : torch_geometric.data.Data or HeteroData
        Graph object to serialize.
    path : Path
        Destination file path (typically under GRAPHS_DATA).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)


def load_graph(path: Path) -> Union[Data, HeteroData]:
    """
    Load a previously saved PyG graph from disk.

    Parameters
    ----------
    path : Path
        Path to a file previously written by save_graph.

    Returns
    -------
    torch_geometric.data.Data or HeteroData

    Notes
    -----
    Uses `weights_only=False`: the file holds a Data/HeteroData object, not a
    plain tensor/state_dict, so torch.load's restricted safe-unpickler
    (weights_only=True) cannot deserialize it.
    """
    return torch.load(path, weights_only=False)
