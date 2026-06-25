# src/utils/graph_io.py
from pathlib import Path
import torch


def save_graph(graph: object, path: Path) -> None:
    """Persist a PyG Data/HeteroData graph to disk, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)


def load_graph(path: Path) -> object:
    """Load a previously saved PyG graph from disk."""
    return torch.load(path, weights_only=False)
