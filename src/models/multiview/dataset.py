# src/models/multiview/dataset.py
"""
MultiviewEventDataset

Converts a cleaned GDELT split DataFrame into tensors for the three views:

  Relational (WHO/WHOM) : actor1_idx, actor2_idx  — indices into the actor graph
  Spatial    (WHERE)    : geo                      — (lat, lon) in degrees
  Temporal   (WHEN)     : time_features            — ℝ × S¹ × S¹ encoding (5-dim)

Temporal features are pre-computed once at dataset construction:
  [0]  t_linear  = (days_since_epoch − μ) / σ     (normalised linear component)
  [1]  sin_year  = sin(2π · doy / 365)             (annual periodicity on S¹)
  [2]  cos_year  = cos(2π · doy / 365)
  [3]  sin_week  = sin(2π · dow / 7)               (weekly periodicity on S¹)
  [4]  cos_week  = cos(2π · dow / 7)

where doy = day-of-year (1–366) and dow = day-of-week (0–6, Mon=0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_REFERENCE_DATE = pd.Timestamp("2000-01-01")
_GEO_LAT_COL = "ActionGeo_Lat"
_GEO_LON_COL = "ActionGeo_Long"
_DAY_COL     = "Day"
_LABEL_COL   = "QuadClass"


def compute_temporal_features(
    day_series: pd.Series,
    linear_mean: float | None = None,
    linear_std: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Compute product-manifold temporal features for a series of YYYYMMDD integers.

    Parameters
    ----------
    day_series : pd.Series of int (YYYYMMDD format)
    linear_mean, linear_std : normalisation statistics.
        If None, computed from day_series (fit on training data).

    Returns
    -------
    features   : np.ndarray of shape (N, 5) float32
    linear_mean, linear_std : statistics used for normalisation
    """
    dates = pd.to_datetime(day_series.astype(str), format="%Y%m%d", errors="coerce")

    linear_day = (dates - _REFERENCE_DATE).dt.days.to_numpy(dtype=np.float32)
    doy = dates.dt.dayofyear.to_numpy(dtype=np.float32)   # 1–366
    dow = dates.dt.dayofweek.to_numpy(dtype=np.float32)   # 0–6

    # Fill NaT with zeros (missing temporal data)
    nan_mask = np.isnan(linear_day)
    linear_day[nan_mask] = 0.0
    doy[nan_mask] = 1.0
    dow[nan_mask] = 0.0

    # Normalise linear component
    if linear_mean is None:
        linear_mean = float(np.nanmean(linear_day))
    if linear_std is None:
        linear_std = float(np.nanstd(linear_day)) + 1e-8

    t_linear = (linear_day - linear_mean) / linear_std

    sin_year = np.sin(2 * np.pi * doy / 365.0).astype(np.float32)
    cos_year = np.cos(2 * np.pi * doy / 365.0).astype(np.float32)
    sin_week = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    cos_week = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

    features = np.stack(
        [t_linear.astype(np.float32), sin_year, cos_year, sin_week, cos_week],
        axis=1,
    )
    return features, linear_mean, linear_std


class MultiviewEventDataset(Dataset):
    """
    Dataset for the MultiviewGeometricModel.

    Parameters
    ----------
    df : cleaned split DataFrame (from SPLITS_DATA parquets)
    actor_to_idx : dict mapping actor string IDs → node indices in actor graph.
        Actors not found are mapped to 0 (unknown slot).
    time_mean, time_std : normalisation statistics for the linear temporal component.
        Pass training-data statistics when constructing val/test datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        actor_to_idx: dict[str, int],
        time_mean: float | None = None,
        time_std: float | None = None,
    ):
        self._len = len(df)

        # --- Relational: actor indices ---
        from src.representation.multiview.actor_graph_builder import _make_actor_ids
        a1_ids = _make_actor_ids(df, "Actor1")
        a2_ids = _make_actor_ids(df, "Actor2")

        self.actor1_idx = torch.tensor(
            pd.Series(a1_ids).map(actor_to_idx).fillna(0).astype(np.int64).values,
            dtype=torch.long,
        )
        self.actor2_idx = torch.tensor(
            pd.Series(a2_ids).map(actor_to_idx).fillna(0).astype(np.int64).values,
            dtype=torch.long,
        )

        # --- Spatial: ActionGeo lat/lon (fill NaN with 0.0) ---
        lat = df[_GEO_LAT_COL].fillna(0.0).to_numpy(dtype=np.float32)
        lon = df[_GEO_LON_COL].fillna(0.0).to_numpy(dtype=np.float32)
        self.geo = torch.tensor(np.stack([lat, lon], axis=1), dtype=torch.float32)

        # --- Temporal: product manifold features ---
        time_feat, self.time_mean, self.time_std = compute_temporal_features(
            df[_DAY_COL], linear_mean=time_mean, linear_std=time_std
        )
        self.time_features = torch.tensor(time_feat, dtype=torch.float32)

        # --- Labels ---
        self.labels = torch.tensor(df[_LABEL_COL].to_numpy(dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        return {
            "actor1_idx":    self.actor1_idx[idx],
            "actor2_idx":    self.actor2_idx[idx],
            "geo":           self.geo[idx],
            "time_features": self.time_features[idx],
            "labels":        self.labels[idx],
        }
