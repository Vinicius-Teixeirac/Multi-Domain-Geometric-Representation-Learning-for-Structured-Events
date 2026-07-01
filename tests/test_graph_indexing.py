"""Tests for deterministic, content-stable node indexing used by both graph builders."""

import pandas as pd
import pytest

from src.representation.graph.indexing import build_index_map, add_node_index


class TestBuildIndexMap:
    def test_assigns_zero_based_sequential_indices(self):
        """Each value gets a zero-based index in the order given (caller must pre-sort)."""
        mapping = build_index_map(["b", "a", "c"])
        assert mapping == {"b": 0, "a": 1, "c": 2}

    def test_sorted_input_gives_stable_alphabetical_order(self):
        """When called with already-sorted input, the resulting indices follow that order."""
        mapping = build_index_map(sorted(["c", "a", "b"]))
        assert mapping == {"a": 0, "b": 1, "c": 2}


class TestAddNodeIndex:
    def test_adds_index_column_in_place(self):
        """The returned mapping and the new DataFrame column must agree, and df is mutated in place."""
        df = pd.DataFrame({"id": ["z", "a", "m", "a"]})
        mapping = add_node_index(df, id_col="id", index_col="node_idx")
        assert "node_idx" in df.columns
        for i, row_id in enumerate(df["id"]):
            assert df["node_idx"].iloc[i] == mapping[row_id]

    def test_index_is_order_independent(self):
        """Indices depend only on the sorted set of unique ids, not on row order."""
        df1 = pd.DataFrame({"id": ["b", "a", "c"]})
        df2 = pd.DataFrame({"id": ["c", "b", "a"]})
        map1 = add_node_index(df1, id_col="id")
        map2 = add_node_index(df2, id_col="id")
        assert map1 == map2

    def test_raises_on_missing_values(self):
        """A NaN in the id column is a data-integrity bug and must raise, not silently index it."""
        df = pd.DataFrame({"id": ["a", None, "b"]})
        with pytest.raises(ValueError):
            add_node_index(df, id_col="id")

    def test_default_index_col_name(self):
        """Without an explicit index_col, the new column must default to 'node_idx'."""
        df = pd.DataFrame({"id": ["a", "b"]})
        add_node_index(df, id_col="id")
        assert "node_idx" in df.columns
