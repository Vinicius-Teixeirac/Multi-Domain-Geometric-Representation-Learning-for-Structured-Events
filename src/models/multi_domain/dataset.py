# src/models/multi_domain/dataset.py
"""
MultiDomainEventDataset

Converts a cleaned GDELT split DataFrame into per-domain tensors:

  Relational (WHO/WHOM) : actor1_idx, actor2_idx  - node indices in actor graph
  Spatial    (WHERE)    : geo                      - (lat, lon) in degrees
                          geo_country_idx          - label-encoded country code
  Temporal   (WHEN)     : time_features            - 3-dim raw temporal features

Temporal feature layout (3-dim) - encoders handle all further transformations:
  [0] t_linear  = (days_since_epoch - mu) / sigma     (normalised linear component)
  [1] doy       = day-of-year  (1 - 366, float)
  [2] dow       = day-of-week  (0 - 6,   float; Mon = 0)

Keeping sin/cos computation out of the dataset allows different temporal
encoders (fixed periods, learnable periods, Fourier) to transform the raw
features however they need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_REFERENCE_DATE  = pd.Timestamp("2000-01-01")
_GEO_LAT_COL     = "ActionGeo_Lat"
_GEO_LON_COL     = "ActionGeo_Long"
_GEO_COUNTRY_COL = "ActionGeo_CountryCode"
_DAY_COL         = "Day"
_LABEL_COL       = "QuadClass"


def compute_temporal_features(
    day_series: pd.Series,
    linear_mean: float | None = None,
    linear_std: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Compute raw temporal features for a series of YYYYMMDD integers.

    Returns a (N, 3) array: [t_linear_normalised, doy, dow].
    Temporal encoders handle all further transformations (sin/cos, Fourier, etc.).

    Parameters
    ----------
    day_series : pd.Series of int (YYYYMMDD format)
    linear_mean, linear_std : normalisation statistics.
        If None, computed from day_series (fit on training split).

    Returns
    -------
    features : np.ndarray of shape (N, 3) float32
    linear_mean, linear_std : statistics used (pass to val/test datasets)
    """
    dates = pd.to_datetime(day_series.astype(str), format="%Y%m%d", errors="coerce")

    linear_day = (dates - _REFERENCE_DATE).dt.days.to_numpy(dtype=np.float32)
    doy = dates.dt.dayofyear.to_numpy(dtype=np.float32)   # 1-366
    dow = dates.dt.dayofweek.to_numpy(dtype=np.float32)   # 0-6

    # Compute stats before filling NaT so artificial zeros don't bias the mean
    if linear_mean is None:
        linear_mean = float(np.nanmean(linear_day))
    if linear_std is None:
        linear_std = float(np.nanstd(linear_day)) + 1e-8

    # Fill NaT with neutral values
    nan_mask = np.isnan(linear_day)
    linear_day[nan_mask] = 0.0
    doy[nan_mask] = 1.0
    dow[nan_mask] = 0.0

    t_linear = ((linear_day - linear_mean) / linear_std).astype(np.float32)

    features = np.stack([t_linear, doy, dow], axis=1)
    return features, linear_mean, linear_std


class MultiDomainEventDataset(Dataset):
    """
    Dataset for the MultiDomainGeometricModel.

    Parameters
    ----------
    df : cleaned split DataFrame (from SPLITS_DATA parquets)
    actor_to_idx : dict mapping actor string IDs -> node indices in actor graph.
        Actors not found are mapped to 0 (unknown slot).
    country_to_idx : dict mapping country code strings -> int indices.
        Pass None to default all country indices to 0 (region-aware encoder
        falls back to zero embedding, other encoders ignore this field).
    time_mean, time_std : normalisation statistics for the linear temporal component.
        Pass training-split statistics when constructing val/test datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        actor_to_idx: dict[str, int],
        country_to_idx: dict[str, int] | None = None,
        time_mean: float | None = None,
        time_std: float | None = None,
    ):
        self._len = len(df)

        # --- Relational: actor indices ---
        from src.representation.multi_domain.actor_graph_builder import _make_actor_ids
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

        # --- Spatial: lat/lon ---
        lat = df[_GEO_LAT_COL].fillna(0.0).to_numpy(dtype=np.float32)
        lon = df[_GEO_LON_COL].fillna(0.0).to_numpy(dtype=np.float32)
        self.geo = torch.tensor(np.stack([lat, lon], axis=1), dtype=torch.float32)

        # --- Spatial: country/region index ---
        if country_to_idx is not None and _GEO_COUNTRY_COL in df.columns:
            codes = df[_GEO_COUNTRY_COL].fillna("").astype(str).values
            idxs = np.array([country_to_idx.get(c, 0) for c in codes], dtype=np.int64)
        else:
            idxs = np.zeros(self._len, dtype=np.int64)
        self.geo_country_idx = torch.tensor(idxs, dtype=torch.long)

        # --- Temporal: raw 3-dim features ---
        time_feat, self.time_mean, self.time_std = compute_temporal_features(
            df[_DAY_COL], linear_mean=time_mean, linear_std=time_std
        )
        self.time_features = torch.tensor(time_feat, dtype=torch.float32)

        # --- Labels ---
        self.labels = torch.tensor(
            df[_LABEL_COL].to_numpy(dtype=np.int64), dtype=torch.long
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        return {
            "actor1_idx":      self.actor1_idx[idx],
            "actor2_idx":      self.actor2_idx[idx],
            "geo":             self.geo[idx],
            "geo_country_idx": self.geo_country_idx[idx],
            "time_features":   self.time_features[idx],
            "labels":          self.labels[idx],
        }
