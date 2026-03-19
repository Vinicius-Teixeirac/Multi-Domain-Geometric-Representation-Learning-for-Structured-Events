# src/representation/graph/heterogeneous/edge_rules.py
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
    Builds bidirectional edges between events and components.

    Returns:
        - event_to_component edge_index
        - component_to_event edge_index (reverse relation)
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
