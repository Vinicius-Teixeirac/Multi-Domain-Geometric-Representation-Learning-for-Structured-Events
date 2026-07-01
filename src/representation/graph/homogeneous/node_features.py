"""Resolves node feature column groups for the homogeneous event graph.

Decides which tabular columns become node features under a given policy,
splitting them into categorical/numeric groups for TabularInputEncoder.
"""

from typing import Dict, List

import pandas as pd

from src.config.schema.columns_schema import COLUMNS_SCHEMA


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def resolve_node_features(
    *,
    policy: str,
    features_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Resolve node feature column groups.

    Policies:
    - "none": no node features (structure-only GNN)
    - "all":  all tabular features (excluding id + target)

    Parameters
    ----------
    policy : str
        "none" or "all" (see policy list above).
    features_df : pd.DataFrame
        Encoded event feature table (output of TabularPipeline) used to
        enumerate available columns when policy == "all".

    Returns:
        {
            "categorical": [...],
            "numeric": [...]
        }

    Notes:
    - ID columns are always excluded
    - Target column is always excluded
    - Node labels are defined elsewhere
    """

    if policy == "none":
        return {"categorical": [], "numeric": []}

    if policy == "all":
        return _all_tabular(features_df)

    raise ValueError(f"Unknown node feature policy '{policy}'.")


# ---------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------
def _all_tabular(features_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Partition all non-id/non-target columns in features_df into categorical and numeric groups."""
    categorical: List[str] = []
    numeric: List[str] = []

    for col in features_df.columns:
        schema = COLUMNS_SCHEMA.get(col)

        # Defensive exclusions
        if schema is not None and schema["kind"] in {"id", "target"}:
            continue

        if schema is not None and schema["kind"] == "categorical":
            categorical.append(col)
        else:
            numeric.append(col)

    return {
        "categorical": categorical,
        "numeric": numeric,
    }
