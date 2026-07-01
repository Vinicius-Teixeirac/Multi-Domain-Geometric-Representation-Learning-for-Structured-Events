"""Tests for shared-key edge construction (homogeneous graph) and event<->component
edge construction (heterogeneous graph)."""

import numpy as np
import pandas as pd
import torch
import pytest

from src.representation.graph.homogeneous.edge_rules import build_binary_edges_from_shared_keys
from src.representation.graph.heterogeneous.edge_rules import build_event_component_edges


class TestBuildBinaryEdgesFromSharedKeys:
    def test_nodes_sharing_a_key_are_connected(self):
        """Two nodes with the same value for a key column must get an edge between them."""
        df = pd.DataFrame({
            "node_idx": [0, 1, 2],
            "actor_id": ["A", "A", "B"],
        })
        src, dst = build_binary_edges_from_shared_keys(
            df, keys=["actor_id"], max_neighbors_per_key=None, default_max_neighbors=10,
        )
        pairs = set(zip(src.tolist(), dst.tolist()))
        assert (0, 1) in pairs and (1, 0) in pairs
        assert (0, 2) not in pairs and (2, 0) not in pairs

    def test_no_self_loops(self):
        """A node must never be connected to itself, even within a shared-key group."""
        df = pd.DataFrame({"node_idx": [0, 1], "actor_id": ["A", "A"]})
        src, dst = build_binary_edges_from_shared_keys(
            df, keys=["actor_id"], max_neighbors_per_key=None, default_max_neighbors=10,
        )
        assert not any(s == d for s, d in zip(src.tolist(), dst.tolist()))

    def test_rows_with_null_key_are_excluded(self):
        """Rows with a missing key value must be dropped before adjacency is computed."""
        df = pd.DataFrame({"node_idx": [0, 1, 2], "actor_id": ["A", None, "A"]})
        src, dst = build_binary_edges_from_shared_keys(
            df, keys=["actor_id"], max_neighbors_per_key=None, default_max_neighbors=10,
        )
        # only nodes 0 and 2 share a non-null key value
        pairs = set(zip(src.tolist(), dst.tolist()))
        assert (0, 2) in pairs
        assert 1 not in src.tolist() and 1 not in dst.tolist()

    def test_neighbor_cap_reduces_edge_density(self):
        """A smaller default_max_neighbors must produce fewer total edges than a larger cap.

        Note: the cap limits how many candidates each node samples *from*, but the
        graph is symmetrized (add_edge adds both directions), so a node's final
        degree can still exceed the cap via edges added by other nodes' independent
        sampling. The reliable, cap-driven signal is total edge count, not per-node
        degree.
        """
        n = 30
        df = pd.DataFrame({"node_idx": list(range(n)), "actor_id": ["A"] * n})
        src_small, _ = build_binary_edges_from_shared_keys(
            df, keys=["actor_id"], max_neighbors_per_key=None, default_max_neighbors=2, seed=0,
        )
        src_large, _ = build_binary_edges_from_shared_keys(
            df, keys=["actor_id"], max_neighbors_per_key=None, default_max_neighbors=25, seed=0,
        )
        assert len(src_small) < len(src_large)

    def test_deterministic_with_fixed_seed(self):
        """The same seed must produce the same sampled edge set (reproducibility)."""
        n = 20
        df = pd.DataFrame({"node_idx": list(range(n)), "actor_id": ["A"] * n})
        src1, dst1 = build_binary_edges_from_shared_keys(
            df, keys=["actor_id"], max_neighbors_per_key=None, default_max_neighbors=3, seed=42,
        )
        src2, dst2 = build_binary_edges_from_shared_keys(
            df, keys=["actor_id"], max_neighbors_per_key=None, default_max_neighbors=3, seed=42,
        )
        assert np.array_equal(src1, src2) and np.array_equal(dst1, dst2)

    def test_dtype_is_int64(self):
        """Edge indices must be int64 to be directly usable as a PyG edge_index tensor."""
        df = pd.DataFrame({"node_idx": [0, 1], "actor_id": ["A", "A"]})
        src, dst = build_binary_edges_from_shared_keys(
            df, keys=["actor_id"], max_neighbors_per_key=None, default_max_neighbors=10,
        )
        assert src.dtype == np.int64 and dst.dtype == np.int64


class TestBuildEventComponentEdges:
    def test_forward_and_reverse_are_transposed(self):
        """The reverse relation edge_index must be the exact transpose of the forward one."""
        df = pd.DataFrame({
            "event_idx": [0, 1, 2],
            "actor_id": ["A", "B", "A"],
        })
        component_index = {"A": 0, "B": 1}
        fwd, rev = build_event_component_edges(
            df, event_idx_col="event_idx", component_col="actor_id", component_index=component_index,
        )
        assert torch.equal(fwd[0], rev[1])
        assert torch.equal(fwd[1], rev[0])

    def test_component_indices_resolved_correctly(self):
        """Destination indices in the forward edge_index must match component_index lookups."""
        df = pd.DataFrame({"event_idx": [0, 1], "actor_id": ["A", "B"]})
        component_index = {"A": 5, "B": 9}
        fwd, _ = build_event_component_edges(
            df, event_idx_col="event_idx", component_col="actor_id", component_index=component_index,
        )
        assert fwd[1].tolist() == [5, 9]

    def test_rows_with_null_component_are_dropped(self):
        """A row with no component value must not produce an edge."""
        df = pd.DataFrame({"event_idx": [0, 1], "actor_id": ["A", None]})
        fwd, _ = build_event_component_edges(
            df, event_idx_col="event_idx", component_col="actor_id", component_index={"A": 0},
        )
        assert fwd.shape[1] == 1

    def test_output_shape_is_two_by_e(self):
        """Both edge_index tensors must have shape (2, E)."""
        df = pd.DataFrame({"event_idx": [0, 1, 2], "actor_id": ["A", "B", "A"]})
        fwd, rev = build_event_component_edges(
            df, event_idx_col="event_idx", component_col="actor_id",
            component_index={"A": 0, "B": 1},
        )
        assert fwd.shape == (2, 3)
        assert rev.shape == (2, 3)
