# src/representation/graph/heterogeneous/node_features.py
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch

from src.config.schema.columns_schema import COLUMNS_SCHEMA
from src.config.paths import ARTIFACTS_DATA
from src.utils.categorical_cardinalities import load_categorical_cardinalities


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def resolve_hetero_node_features(
    *,
    policy: str,
    features_df: pd.DataFrame,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Resolve node feature columns for heterogeneous graphs.

    Policies:
    - "none":
        No node features for any node type.
    - "all":
        Event nodes receive all tabular features
        (excluding id + target). Other node types receive none.

    Returns:
        {
            "event": {
                "categorical": [...],
                "numeric": [...]
            }
        }

    Notes:
    - Only 'event' nodes are eligible for features
    - Other node types are intentionally feature-less
    """

    if policy == "none":
        return {}

    if policy == "all":
        return {
            "event": _all_event_features(features_df)
        }

    raise ValueError(f"Unknown node feature policy '{policy}'.")


# ---------------------------------------------------------------------
# Tensor builder
# ---------------------------------------------------------------------

def build_event_node_tensors(
    *,
    features_df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    dataset_name: str,
    split_tag: str = "default",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Optional[torch.Tensor]]:
    """
    Build event node feature tensors and metadata.

    Returns:
        x_cat: Dict[str, LongTensor]        (per-column categorical features)
        cat_cardinalities: Dict[str, int]  (per-column vocab sizes)
        x_num: FloatTensor | None           (stacked numeric features)
    """

    # --------------------------------------------------
    # Categorical features
    # --------------------------------------------------
    x_cat: Dict[str, torch.Tensor] = {}
    if categorical_cols:
        for col in categorical_cols:
            x_cat[col] = torch.from_numpy(
                features_df[col].to_numpy(dtype=np.int64)
            )

    # --------------------------------------------------
    # Load categorical cardinalities (from artifacts)
    # --------------------------------------------------
    artifacts_dir = (
        ARTIFACTS_DATA
        / dataset_name
        / "features"
        / split_tag
    )

    cat_cardinalities = load_categorical_cardinalities(
        categorical_cols=categorical_cols,
        artifacts_dir=artifacts_dir,
    )

    # --------------------------------------------------
    # Numeric features (stacked)
    # --------------------------------------------------
    if numeric_cols:
        x_num = torch.from_numpy(
            features_df[numeric_cols].to_numpy(dtype=np.float32)
        )
    else:
        x_num = None

    return x_cat, cat_cardinalities, x_num


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _all_event_features(features_df: pd.DataFrame) -> Dict[str, List[str]]:
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
