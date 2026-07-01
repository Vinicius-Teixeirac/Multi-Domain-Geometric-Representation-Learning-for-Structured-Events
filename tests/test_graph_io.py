"""Tests for PyG graph serialization helpers."""

import torch
from torch_geometric.data import Data

from src.utils.graph_io import save_graph, load_graph


class TestSaveLoadGraph:
    def test_round_trip_preserves_data(self, tmp_path):
        """A graph saved and reloaded must have identical tensors to the original."""
        graph = Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            y=torch.tensor([0, 1, 0, 1, 0]),
        )
        path = tmp_path / "nested" / "graph.pt"
        save_graph(graph, path)
        loaded = load_graph(path)

        assert torch.equal(loaded.x, graph.x)
        assert torch.equal(loaded.edge_index, graph.edge_index)
        assert torch.equal(loaded.y, graph.y)

    def test_creates_parent_directories(self, tmp_path):
        """save_graph must create missing parent directories rather than raising."""
        graph = Data(x=torch.randn(2, 2), edge_index=torch.tensor([[0], [1]]))
        path = tmp_path / "a" / "b" / "c" / "graph.pt"
        save_graph(graph, path)
        assert path.exists()
