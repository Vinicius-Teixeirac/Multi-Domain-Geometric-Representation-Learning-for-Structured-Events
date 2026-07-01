"""Vectorized construction of event<->component bidirectional edge indices.

Used by heterogeneous/builder.py to link event nodes to each of their
actor1/actor2/geo/day component nodes without a per-row Python loop.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple


def build_event_component_edges(
    df: pd.DataFrame,
    *,
    event_idx_col: str,
    component_col: str,
    component_index: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build bidirectional edges between events and one component type.

    Rows with a null component value are dropped (that event simply gets no
    edge for this relation) rather than raising, since not every event has
    every component populated.

    Parameters
    ----------
    df : pd.DataFrame
        Entity table containing at least `event_idx_col` and `component_col`.
    event_idx_col : str
        Column holding the integer event node index (see indexing.add_node_index).
    component_col : str
        Column holding the raw component entity ID (e.g. "Actor1ID").
    component_index : dict
        Mapping from raw component ID to its integer node index within this split.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor), each of shape (2, E)
        (event_to_component edge_index, component_to_event edge_index),
        i.e. the forward relation and its reverse.
    """

    # --------------------------------------------------
    # Select + clean
    # --------------------------------------------------
    sub = df[[event_idx_col, component_col]].dropna()

    # --------------------------------------------------
    # Vectorized index resolution
    # --------------------------------------------------
    src_e = sub[event_idx_col].to_numpy(dtype=np.int64)
    dst_c = sub[component_col].map(component_index).to_numpy(dtype=np.int64)

    # --------------------------------------------------
    # FAST tensor construction (no warning, no copies)
    # --------------------------------------------------
    edge_event_to_comp = torch.from_numpy(
        np.stack((src_e, dst_c), axis=0)
    )

    edge_comp_to_event = torch.from_numpy(
        np.stack((dst_c, src_e), axis=0)
    )

    return edge_event_to_comp, edge_comp_to_event
