# src/representation/tabular/transformers.py
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)

class IntegerDateParser:
    def __init__(self, fmt: str):
        self.fmt = fmt
        

    def fit(self, X):
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        logger.debug(f"Parsing integer date column '{X.name}' with format '{self.fmt}'")
        return pd.to_datetime(X.astype(str), format=self.fmt)


class DateToCyclic:
    def __init__(self, period: int):
        self.period = period

    def fit(self, X):
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
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
    def __init__(self, prefix: str):
        self.prefix = prefix
        logger.debug(f"Converting geographic coordinates to Cartesian features with prefix {prefix}")

    def fit(self, X):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Input columns: {list(X.columns)}")
        lat = np.radians(X.iloc[:, 0])
        lon = np.radians(X.iloc[:, 1])

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
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: pd.Series):
        self.scaler.fit(X.to_numpy().reshape(-1, 1))
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        values = self.scaler.transform(X.to_numpy().reshape(-1, 1)).ravel()
        return pd.Series(values, name=X.name, index=X.index)

    def save(self, path: Path):
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
        data = json.loads(path.read_text())
        obj = cls()
        obj.scaler.mean_ = np.array(data["mean"])
        obj.scaler.scale_ = np.array(data["scale"])
        obj.scaler.var_ = np.array(data["var"])
        obj.scaler.n_features_in_ = data["n_features"]
        return obj
