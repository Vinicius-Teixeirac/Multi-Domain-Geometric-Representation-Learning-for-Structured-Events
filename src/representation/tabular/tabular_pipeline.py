# src/representation/tabular/tabular_pipeline.py
from __future__ import annotations

import warnings
from typing import Dict, List

import pandas as pd

from src.config.paths import (
    SPLITS_DATA,
    ARTIFACTS_DATA,
    FEATURES_DATA,
)
from src.config.schema.columns_schema import COLUMNS_SCHEMA
from src.config.schema.encoding_schema import ENCODING_SCHEMA
from src.representation.tabular.encoding import (
    SafeLabelEncoder,
    HashEncoder,
)
from src.representation.tabular.transformers import (
    IntegerDateParser,
    DateToCyclic,
    GeoToCartesian,
    StandardScalerWrapper,
)
from src.utils.loading import load_parquet
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)


class TabularPipeline:
    """
    Schema-driven tabular feature encoding pipeline.

    Responsibilities:
    - Validate columns against COLUMNS_SCHEMA and ENCODING_SCHEMA
    - Fit encoders on train split only
    - Transform train / valid / test deterministically
    - Persist encoder artifacts
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, dataset_name: str, split_tag: str = "default"):
        self.dataset_name = dataset_name
        self.split_tag = split_tag
        self.input_dir = SPLITS_DATA / dataset_name
        self.artifacts_dir = ARTIFACTS_DATA / dataset_name / "features" / split_tag
        self.output_dir = FEATURES_DATA / dataset_name

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fitted_objects: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("Loading splits...")
        train = self._load_split("train")
        test = self._load_split("test")

        try:
            valid = self._load_split("valid")
        except FileNotFoundError:
            valid = None

        logger.info("Validating schemas...")
        self._validate_columns(train)

        logger.info("Fitting encoders on train split...")
        X_train = self._fit_transform(train)

        logger.info("Transforming validation / test splits...")
        X_valid = self._transform(valid) if valid is not None else None
        X_test = self._transform(test)

        logger.info("Saving encoded features...")
        self._save_split(X_train, "train")
        if X_valid is not None:
            self._save_split(X_valid, "valid")
        self._save_split(X_test, "test")

        logger.info("Tabular pipeline completed successfully.")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_split(self, split: str) -> pd.DataFrame:
        return load_parquet(f"{split}_{self.split_tag}.parquet", self.input_dir)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_columns(self, df: pd.DataFrame) -> None:
        schema_cols = set(COLUMNS_SCHEMA.keys())
        df_cols = set(df.columns)

        missing = schema_cols - df_cols
        extra = df_cols - schema_cols

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if extra:
            warnings.warn(
                f"Dataset contains unexpected columns that will be ignored: "
                f"{sorted(extra)}"
            )

        for col, enc_cfg in ENCODING_SCHEMA.items():
            if col not in COLUMNS_SCHEMA:
                raise ValueError(
                    f"Encoding schema references unknown column '{col}'."
                )

            kind = COLUMNS_SCHEMA[col]["kind"]
            method = enc_cfg["method"]
            self._validate_encoding_compatibility(col, kind, method)

    @staticmethod
    def _validate_encoding_compatibility(
        col: str, kind: str, method: str
    ) -> None:
        valid = {
            "categorical": {"label", "hash"},
            "date": {"cyclical"},
            "geo": {"geodetic_cartesian"},
        }

        if kind in valid and method not in valid[kind]:
            raise ValueError(
                f"Incompatible encoding: column '{col}' "
                f"(kind={kind}) cannot use method '{method}'."
            )

    # ------------------------------------------------------------------
    # Core transform logic
    # ------------------------------------------------------------------
    def _fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        for col in df.columns:
            kind = COLUMNS_SCHEMA[col]["kind"]

            if kind in {"id", "target"}:
                frames.append(df[[col]])
                continue

            # Geo handled once per pair (Lat triggers, Long skipped)
            if kind == "geo" and col.endswith("_Long"):
                continue

            frames.append(self._fit_transform_column(df, col))

        return pd.concat(frames, axis=1)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        for col in df.columns:
            kind = COLUMNS_SCHEMA[col]["kind"]

            if kind in {"id", "target"}:
                frames.append(df[[col]])
                continue

            if kind == "geo" and col.endswith("_Long"):
                continue

            frames.append(self._transform_column(df, col))

        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # Column-level logic
    # ------------------------------------------------------------------
    def _fit_transform_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        cfg = ENCODING_SCHEMA.get(col)
        series = df[col]

        if cfg is None:
            return series.to_frame(col)

        method = cfg["method"]
        params = cfg.get("params", {})

        logger.info(f"Fitting encoder for column '{col}' using '{method}'")

        if method == "label":
            enc = SafeLabelEncoder().fit(series)
            self._save_artifact(col, enc)
            return enc.transform(series).to_frame(col)

        if method == "hash":
            enc = HashEncoder(num_buckets=params["hash_dim"])
            self._save_artifact(col, enc)
            return enc.transform(series).to_frame(col)

        if method == "cyclical":
            parser = IntegerDateParser(fmt="%Y%m%d")
            dates = parser.transform(series)
            cyc = DateToCyclic(period=params["period"])
            return cyc.transform(dates.dt.dayofyear)

        if method == "geodetic_cartesian":
            if not col.endswith("_Lat"):
                return pd.DataFrame(index=df.index)

            prefix = col.replace("_Lat", "")
            lat_col = f"{prefix}_Lat"
            lon_col = f"{prefix}_Long"

            geo = GeoToCartesian(prefix)
            out = geo.transform(df[[lat_col, lon_col]])

            if params.get("scale", False):
                for c in out.columns:
                    scaler = StandardScalerWrapper().fit(out[c])
                    self._save_artifact(c, scaler)
                    out[c] = scaler.transform(out[c])

            return out

        raise ValueError(f"Unknown encoding method '{method}' for column '{col}'.")

    def _transform_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        cfg = ENCODING_SCHEMA.get(col)
        series = df[col]

        if cfg is None:
            return series.to_frame(col)

        method = cfg["method"]
        params = cfg.get("params", {})

        if method in {"label", "hash"}:
            enc = self._load_artifact(col)
            return enc.transform(series).to_frame(col)

        if method == "cyclical":
            parser = IntegerDateParser(fmt="%Y%m%d")
            dates = parser.transform(series)
            cyc = DateToCyclic(period=params["period"])
            return cyc.transform(dates.dt.dayofyear)

        if method == "geodetic_cartesian":
            if not col.endswith("_Lat"):
                return pd.DataFrame(index=df.index)

            prefix = col.replace("_Lat", "")
            lat_col = f"{prefix}_Lat"
            lon_col = f"{prefix}_Long"

            geo = GeoToCartesian(prefix)
            out = geo.transform(df[[lat_col, lon_col]])

            if params.get("scale", False):
                for c in out.columns:
                    scaler = self._load_artifact(c)
                    out[c] = scaler.transform(out[c])

            return out

        raise ValueError(f"Unknown encoding method '{method}' for column '{col}'.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_artifact(self, name: str, obj) -> None:
        path = self.artifacts_dir / f"{name}.json"
        obj.save(path)
        self.fitted_objects[name] = obj
        logger.debug(f"Saved artifact '{name}' to {path}")

    def _load_artifact(self, name: str):
        if name in self.fitted_objects:
            return self.fitted_objects[name]

        path = self.artifacts_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact for '{name}'.")

        return StandardScalerWrapper.load(path) \
            if name not in ENCODING_SCHEMA else (
                SafeLabelEncoder.load(path)
                if ENCODING_SCHEMA[name]["method"] == "label"
                else HashEncoder.load(path)
            )

    def _save_split(self, df: pd.DataFrame, split: str) -> None:
        out = self.output_dir / f"{split}_{self.split_tag}_features.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Saved {split} features with shape {df.shape} to {out}")
