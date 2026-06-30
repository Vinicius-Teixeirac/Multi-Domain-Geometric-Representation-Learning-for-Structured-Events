# src/representation/graph/indexing.py

from typing import Dict
import pandas as pd


def build_index_map(values) -> Dict:
    """
    Build a stable {value: integer_index} mapping from an iterable.

    Parameters
    ----------
    values : iterable
        Sequence of unique values (caller must ensure they are already sorted
        for deterministic ordering).

    Returns
    -------
    dict
        Mapping from each value to its zero-based index.
    """
    return {v: i for i, v in enumerate(values)}


def add_node_index(
    df: pd.DataFrame,
    id_col: str,
    index_col: str = "node_idx",
) -> Dict:
    """
    Add a deterministic, content-stable node index column to df in-place.

    The index is split-local and independent of row order: unique values in
    id_col are sorted, then assigned consecutive integers starting at 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to annotate; modified in-place.
    id_col : str
        Column whose unique values form the node vocabulary.
    index_col : str
        Name of the new integer index column to create.

    Returns
    -------
    dict
        Mapping from each unique id_col value to its assigned integer index.
    """
    if df[id_col].isna().any():
        raise ValueError(f"Missing values found in id column '{id_col}'")

    # stable ordering independent of row order
    ids = pd.unique(df[id_col])
    ids = sorted(ids)

    mapping = build_index_map(ids)
    df[index_col] = df[id_col].map(mapping)

    return mapping
