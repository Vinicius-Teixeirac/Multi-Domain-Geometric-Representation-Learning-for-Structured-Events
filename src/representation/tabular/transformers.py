# src/representation/tabular/transformers.py
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)

class IntegerDateParser:
    """Parse an integer date column (e.g. 20150923) into a pandas datetime Series."""

    def __init__(self, fmt: str):
        """
        Parameters
        ----------
        fmt : str
            strptime format string for parsing (e.g. '%Y%m%d').
        """
        self.fmt = fmt

    def fit(self, X):
        """No-op; returns self for API uniformity."""
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Convert an integer date Series to pandas datetime using self.fmt."""
        logger.debug(f"Parsing integer date column '{X.name}' with format '{self.fmt}'")
        return pd.to_datetime(X.astype(str), format=self.fmt)


class DateToCyclic:
    """Map a numeric day-of-year (or similar periodic) series to sin/cos features."""

    def __init__(self, period: int):
        """
        Parameters
        ----------
        period : int
            The full cycle length (e.g. 365 for annual, 7 for weekly).
        """
        self.period = period

    def fit(self, X):
        """No-op; returns self for API uniformity."""
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        """
        Return a two-column DataFrame with sin and cos cyclic features.

        Parameters
        ----------
        X : pd.Series
            Numeric values within [1, period].

        Returns
        -------
        pd.DataFrame with columns '{X.name}_sin' and '{X.name}_cos'.
        """
        values = X.astype(float)

        sin = np.sin(2 * np.pi * values / self.period)
        cos = np.cos(2 * np.pi * values / self.period)

        logger.debug(f"Converting date column '{X.name}' to cyclic features (period={self.period})")
        return pd.DataFrame(
            {
                f"{X.name}_sin": sin,
                f"{X.name}_cos": cos,
            },
            index=X.index,
        )

class GeoToCartesian:
    """Convert (lat, lon) degree columns to Earth-centered Cartesian (x, y, z) coordinates."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        logger.debug(f"Converting geographic coordinates to Cartesian features with prefix {prefix}")

    def fit(self, X):
        """No-op; returns self for API uniformity."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert (lat, lon) degree columns to Cartesian (x, y, z) on a unit sphere.

        Parameters
        ----------
        X : pd.DataFrame
            Two-column DataFrame; first column is latitude, second is longitude
            (both in degrees).

        Returns
        -------
        pd.DataFrame with columns '{prefix}_geo_x', '{prefix}_geo_y', '{prefix}_geo_z'.
        """
        logger.debug(f"Input columns: {list(X.columns)}")
        lat = np.radians(X.iloc[:, 0])
        lon = np.radians(X.iloc[:, 1])

        # Standard spherical-to-Cartesian conversion on a unit sphere
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        return pd.DataFrame(
            {
                f"{self.prefix}_geo_x": x,
                f"{self.prefix}_geo_y": y,
                f"{self.prefix}_geo_z": z,
            },
            index=X.index,
        )


class StandardScalerWrapper:
    """Thin sklearn StandardScaler wrapper with JSON serialisation for pipeline persistence."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: pd.Series):
        """Fit the StandardScaler on X; return self."""
        self.scaler.fit(X.to_numpy().reshape(-1, 1))
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Standardise X using the fitted scaler; return a Series with the same name and index."""
        values = self.scaler.transform(X.to_numpy().reshape(-1, 1)).ravel()
        return pd.Series(values, name=X.name, index=X.index)

    def save(self, path: Path):
        """Serialise scaler parameters (mean, scale, var) to a JSON file at path."""
        path.write_text(
            json.dumps(
                {
                    "mean": self.scaler.mean_.tolist(),
                    "scale": self.scaler.scale_.tolist(),
                    "var": self.scaler.var_.tolist(),
                    "n_features": int(self.scaler.n_features_in_),
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "StandardScalerWrapper":
        """Restore a previously saved StandardScalerWrapper from a JSON file."""
        data = json.loads(path.read_text())
        obj = cls()
        obj.scaler.mean_ = np.array(data["mean"])
        obj.scaler.scale_ = np.array(data["scale"])
        obj.scaler.var_ = np.array(data["var"])
        obj.scaler.n_features_in_ = data["n_features"]
        return obj
