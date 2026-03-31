# src/representation/multiview/actor_graph_builder.py
"""
Builds the actor-actor co-occurrence graph from GDELT split DataFrames.

Design:
  - Nodes  = unique actor IDs (string keys derived from 8 actor attribute columns)
  - Edges  = co-occurrence in the same training event (undirected)
  - Node features = label-encoded categorical actor attributes (int64, one per attribute)
  - Index 0 is reserved as "unknown / padding" actor

The GNN encoder inside the model learns embeddings from these features via
message-passing, capturing relational structure between actors.

NOTE: For very large datasets (>1M events, >500K unique actors) consider
replacing the full-graph GNN with neighbor sampling (PyG NeighborLoader).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

# -----------------------------------------------------------------------
# Actor attribute columns (same schema for Actor1 and Actor2)
# -----------------------------------------------------------------------
_ATTR_BASES: list[str] = [
    "CountryCode",
    "KnownGroupCode",
    "EthnicCode",
    "Religion1Code",
    "Religion2Code",
    "Type1Code",
    "Type2Code",
    "Type3Code",
]

_NUM_ATTRS = len(_ATTR_BASES)
_UNKNOWN_TOKEN = "__unk__"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _actor_cols(prefix: str) -> list[str]:
    return [f"{prefix}{b}" for b in _ATTR_BASES]


def _make_actor_ids(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Concatenate actor attribute columns into a unique string ID per row."""
    cols = _actor_cols(prefix)
    str_cols = [df[c].fillna("").astype(str) for c in cols]
    return (str_cols[0] + "-" + str_cols[1] + "-" + str_cols[2] + "-" +
            str_cols[3] + "-" + str_cols[4] + "-" + str_cols[5] + "-" +
            str_cols[6] + "-" + str_cols[7]).values


def _safe_transform(values: np.ndarray, enc: LabelEncoder) -> np.ndarray:
    """
    Label-encode an array of values with +1 offset (index 0 = unknown).
    Values not seen during fitting are mapped to 0.
    """
    classes_set = set(enc.classes_)
    out = np.zeros(len(values), dtype=np.int64)
    mask = np.array([v in classes_set for v in values], dtype=bool)
    if mask.any():
        out[mask] = enc.transform(np.asarray(values)[mask]) + 1
    return out


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------
def build_actor_graph(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
) -> tuple[Data, dict[str, int], list[int], list[LabelEncoder]]:
    """
    Build an undirected actor co-occurrence graph.

    All actor IDs across ALL available splits are registered as nodes so that
    val/test actors are not "unseen" at inference (transductive actor setting,
    inductive event setting — consistent with the existing codebase).
    Only training events contribute edges.

    Parameters
    ----------
    df_train : cleaned training-split DataFrame (from splits/ parquet)
    df_val, df_test : optional val/test splits for actor node registration

    Returns
    -------
    graph : PyG Data
        x          : (N_actors, 8)  — int64 label-encoded actor attributes
        edge_index : (2, E)         — undirected co-occurrence edges (train only)
        num_nodes  : N_actors
    actor_to_idx : dict[str, int]
        Maps actor string ID → node index. "__unknown__" → 0.
    cardinalities : list[int]
        Per-feature cardinality including +1 for the "unknown" slot.
        Use these to size nn.Embedding layers in the model.
    encoders : list[LabelEncoder]
        One fitted encoder per attribute column (8 total).
    """
    # ------------------------------------------------------------------
    # 1. Collect actor IDs across all available splits
    # ------------------------------------------------------------------
    dfs = [df_train] + ([df_val] if df_val is not None else []) + ([df_test] if df_test is not None else [])
    full = pd.concat(dfs, ignore_index=True)

    a1_ids_full = _make_actor_ids(full, "Actor1")
    a2_ids_full = _make_actor_ids(full, "Actor2")
    all_actor_ids = np.unique(np.concatenate([a1_ids_full, a2_ids_full]))

    # Index 0 reserved for unknown
    actor_to_idx: dict[str, int] = {"__unknown__": 0}
    for i, aid in enumerate(all_actor_ids, start=1):
        actor_to_idx[aid] = i
    N = len(actor_to_idx)  # total nodes including unknown slot

    # ------------------------------------------------------------------
    # 2. Fit label encoders on combined values from all splits
    # ------------------------------------------------------------------
    encoders: list[LabelEncoder] = []
    cardinalities: list[int] = []
    for base in _ATTR_BASES:
        col1, col2 = f"Actor1{base}", f"Actor2{base}"
        combined = (
            pd.concat([full[col1], full[col2]])
            .fillna(_UNKNOWN_TOKEN)
            .astype(str)
        )
        le = LabelEncoder()
        le.fit(combined)
        encoders.append(le)
        cardinalities.append(len(le.classes_) + 1)  # +1 for unknown at embedding index 0

    # ------------------------------------------------------------------
    # 3. Build node feature matrix  (N, 8)
    # ------------------------------------------------------------------
    # Collect one representative attribute row per unique actor
    a1_attr = {b: full[f"Actor1{b}"].fillna(_UNKNOWN_TOKEN).astype(str).values for b in _ATTR_BASES}
    a2_attr = {b: full[f"Actor2{b}"].fillna(_UNKNOWN_TOKEN).astype(str).values for b in _ATTR_BASES}

    a1_df = pd.DataFrame({"actor_id": a1_ids_full, **a1_attr})
    a2_df = pd.DataFrame({"actor_id": a2_ids_full, **a2_attr})
    actor_attr_df = (
        pd.concat([a1_df, a2_df], ignore_index=True)
        .groupby("actor_id", sort=False)
        .first()
        .reset_index()
    )

    # Map actor IDs to node indices (vectorised)
    node_indices = (
        pd.Series(actor_attr_df["actor_id"].values)
        .map(actor_to_idx)
        .fillna(0)
        .astype(np.int64)
        .values
    )

    node_x = np.zeros((N, _NUM_ATTRS), dtype=np.int64)
    for feat_i, (base, enc) in enumerate(zip(_ATTR_BASES, encoders)):
        encoded = _safe_transform(actor_attr_df[base].values, enc)
        node_x[node_indices, feat_i] = encoded

    # ------------------------------------------------------------------
    # 4. Build edges from training data only  (undirected)
    # ------------------------------------------------------------------
    train_a1_ids = _make_actor_ids(df_train, "Actor1")
    train_a2_ids = _make_actor_ids(df_train, "Actor2")

    a1_idx_arr = (
        pd.Series(train_a1_ids).map(actor_to_idx).fillna(0).astype(np.int64).values
    )
    a2_idx_arr = (
        pd.Series(train_a2_ids).map(actor_to_idx).fillna(0).astype(np.int64).values
    )

    mask = a1_idx_arr != a2_idx_arr  # remove self-loops
    src = np.concatenate([a1_idx_arr[mask], a2_idx_arr[mask]])
    dst = np.concatenate([a2_idx_arr[mask], a1_idx_arr[mask]])

    if src.size > 0:
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)  # deduplicate
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    graph = Data(
        x=torch.tensor(node_x, dtype=torch.long),
        edge_index=edge_index,
        num_nodes=N,
    )
    return graph, actor_to_idx, cardinalities, encoders
