# src/representation/tabular/encoding.py
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from scipy import sparse

from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Safe Label Encoder
# ---------------------------------------------------------------------
class SafeLabelEncoder:
    """
    Deterministic label encoder with explicit UNK handling.

    - UNK token = 0
    - Known categories start from 1
    - Must be fitted before transform
    """

    def __init__(self, dtype: str = "int32"):
        self.mapping: dict[Any, int] = {}
        self.unk_token = 0
        self.num_classes_: int = 0
        self._fitted: bool = False
        self.dtype = dtype

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, series: pd.Series) -> "SafeLabelEncoder":
        if self._fitted:
            raise RuntimeError("SafeLabelEncoder is already fitted.")
        uniques = sorted(series.dropna().unique())
        self.mapping = {v: i for i, v in enumerate(uniques, start=1)}
        self.num_classes_ = len(self.mapping)
        self._fitted = True
        logger.info(f"Fitted SafeLabelEncoder for column '{series.name}' with {self.num_classes_} categories (+UNK)")
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("SafeLabelEncoder must be fitted before transform().")

        return (
            series.map(self.mapping)
            .fillna(self.unk_token)
            .astype(self.dtype)
            .rename(series.name)
        )

    def save(self, path: Path):
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted SafeLabelEncoder.")

        path.write_text(
            json.dumps(
                {
                    "mapping": self.mapping,
                    "dtype": self.dtype,
                }
            )
        )
        logger.debug(f"Saved SafeLabelEncoder to {path}")


    @classmethod
    def load(cls, path: Path) -> "SafeLabelEncoder":
        data = json.loads(path.read_text())
        obj = cls(dtype=data.get("dtype", "int32"))
        obj.mapping = data["mapping"]
        obj.num_classes_ = len(obj.mapping)
        obj._fitted = True
        return obj


# ---------------------------------------------------------------------
# Scalar Hash Encoder (Embedding-friendly)
# ---------------------------------------------------------------------
class HashEncoder:
    """
    Scalar hashing encoder.

    Produces integer bucket IDs suitable for embedding layers.
    Hash collisions are intentionally resolved by downstream
    representation learning.
    """

    def __init__(self, num_buckets: int, dtype: str = "int32"):
        self.num_buckets = num_buckets
        self.dtype = dtype
        logger.info(f"Initialized HashEncoder with {self.num_buckets} buckets")

    @property
    def is_fitted(self) -> bool:
        # Stateless by design
        return True

    def fit(self, series: pd.Series) -> "HashEncoder":
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        values = (
            series.fillna("__nan__")
            .astype(str)
            .apply(
                lambda x: int(
                    hashlib.md5(x.encode("utf-8")).hexdigest(), 16
                )
                % self.num_buckets
            )
            .astype(self.dtype)
        )
        values.name = series.name
        logger.debug(f"Hash-encoding column '{series.name}' into {self.num_buckets} buckets")
        return values

    def save(self, path: Path):
        path.write_text(
            json.dumps(
                {
                    "num_buckets": self.num_buckets,
                    "dtype": self.dtype,
                }
            )
        )
        logger.debug(f"Saved HashEncoder to {path}")


    @classmethod
    def load(cls, path: Path) -> "HashEncoder":
        data = json.loads(path.read_text())
        return cls(
            num_buckets=data["num_buckets"],
            dtype=data.get("dtype", "int32"),
        )


# ---------------------------------------------------------------------
# Hashed One-Hot Encoder
# ---------------------------------------------------------------------
class HashedOneHotEncoder:
    """
    Fixed, non-trainable, feature-hashing one-hot encoder.

    Intended ONLY for:
    - linear models
    - architecture-only inductive bias experiments

    WARNING:
    Large num_buckets can cause severe memory usage.
    """

    def __init__(
        self,
        num_buckets: int,
        signed: bool = True,
        dtype: np.dtype = np.float32,
    ):
        self.num_buckets = num_buckets
        self.signed = signed
        self.dtype = dtype

        if self.num_buckets > 100_000:
            logger.warning(
                f"HashedOneHotEncoder initialized with {self.num_buckets} buckets. "
                "This may cause memory and performance issues."
            )

    @property
    def is_fitted(self) -> bool:
        # Stateless by design
        return True

    def fit(self, series: pd.Series) -> "HashedOneHotEncoder":
        return self

    def _hash(self, value: str) -> int:
        return (
            int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)
            % self.num_buckets
        )

    def _sign(self, value: str) -> float:
        if not self.signed:
            return 1.0
        return 1.0 if (hash(value) & 1) == 0 else -1.0

    def transform(self, series: pd.Series) -> sparse.csr_matrix:
        series = series.fillna("__nan__").astype(str)

        n_samples = len(series)
        rows = np.arange(n_samples, dtype=np.int32)

        cols = np.fromiter(
            (self._hash(v) for v in series),
            dtype=np.int32,
            count=n_samples,
        )

        data = np.fromiter(
            (self._sign(v) for v in series),
            dtype=self.dtype,
            count=n_samples,
        )
        logger.debug(f"Applying HashedOneHotEncoder to column '{series.name}' (output shape: {n_samples} × {self.num_buckets})")
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_samples, self.num_buckets),
            dtype=self.dtype,
        )

    def save(self, path: Path):
        path.write_text(
            json.dumps(
                {
                    "num_buckets": self.num_buckets,
                    "signed": self.signed,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "HashedOneHotEncoder":
        data = json.loads(path.read_text())
        return cls(
            num_buckets=data["num_buckets"],
            signed=data.get("signed", True),
        )
