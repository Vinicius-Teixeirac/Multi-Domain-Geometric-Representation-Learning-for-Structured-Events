# src/representation/graph/indexing.py

from typing import Dict
import pandas as pd


def build_index_map(values) -> Dict:
    """
    Builds a stable index mapping from values to integers.
    Assumes values are already sorted.
    """
    return {v: i for i, v in enumerate(values)}


def add_node_index(
    df: pd.DataFrame,
    id_col: str,
    index_col: str = "node_idx",
) -> Dict:
    """
    Adds a deterministic node index column to the dataframe.

    Indexing is:
        - split-local
        - stable w.r.t. content (not row order)
    """
    if df[id_col].isna().any():
        raise ValueError(f"Missing values found in id column '{id_col}'")

    # stable ordering independent of row order
    ids = pd.unique(df[id_col])
    ids = sorted(ids)

    mapping = build_index_map(ids)
    df[index_col] = df[id_col].map(mapping)

    return mapping
