# src/utils/graph_io.py
from pathlib import Path
import torch


def save_graph(graph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)


def load_graph(path: Path):
    return torch.load(path, weights_only=False)
