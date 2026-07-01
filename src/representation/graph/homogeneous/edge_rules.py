"""Shared-key edge construction for the homogeneous event graph.

Two event nodes are connected if they share at least one entity key value
(e.g. the same Actor1ID), with per-key neighbour sampling to bound degree.
"""

from typing import Dict, Set, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

def build_binary_edges_from_shared_keys(
    df: pd.DataFrame,
    keys: Iterable[str],
    *,
    node_idx_col: str = "node_idx",
    max_neighbors_per_key: Optional[Dict[str, int]],
    default_max_neighbors: int = 10,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an undirected binary graph where two nodes are connected if they share at least one key.

    Neighbour sampling is applied per key per node to cap the edge density.
    Duplicate edges (same pair via multiple keys) are eliminated so the
    result is a simple graph.

    Parameters
    ----------
    df : pd.DataFrame
        Entity DataFrame with a node index column and one column per key.
    keys : iterable of str
        Column names used to define shared-key adjacency.
    node_idx_col : str
        Column containing integer node indices.
    max_neighbors_per_key : dict or None
        Per-key neighbour sampling limit. Falls back to default_max_neighbors.
    default_max_neighbors : int
        Default per-node neighbour cap when the key has no explicit limit.
    seed : int or None
        Seed for the random neighbour sampler (None = non-deterministic).

    Returns
    -------
    src : np.ndarray of shape (E,) int64
    dst : np.ndarray of shape (E,) int64
        Source and destination node indices of the undirected edge set.
    """
    rng = np.random.default_rng(seed)

    # adjacency as sets -> binary by construction
    neighbors: Dict[int, Set[int]] = {}

    def add_edge(u: int, v: int):
        """Add an undirected edge between u and v, skipping self-loops."""
        if u == v:
            return
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    for key in keys:
        limit = max_neighbors_per_key.get(key, default_max_neighbors) if max_neighbors_per_key else default_max_neighbors
        grouped = (
            df[[key, node_idx_col]]
            .dropna()
            .groupby(key, sort=False)
        )

        for _, group in grouped:
            idx = group[node_idx_col].to_numpy(dtype=np.int64)
            n = idx.size

            if n < 2:
                continue

            for i, v in enumerate(idx):
                candidates = np.delete(idx, i)

                if candidates.size == 0:
                    continue

                if candidates.size > limit:
                    sampled = rng.choice(
                        candidates,
                        size=limit,
                        replace=False,
                    )
                else:
                    sampled = candidates

                for u in sampled:
                    add_edge(v, u)

    # materialize edge_index
    src, dst = [], []
    for u, nbrs in neighbors.items():
        for v in nbrs:
            src.append(u)
            dst.append(v)

    return (
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
    )
