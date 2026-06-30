# src/preprocessing/cleaning.py

from pathlib import Path
from typing import Iterable, Optional, List
import warnings

import pandas as pd

from src.config.paths import RAW_DATA, PROCESSED_DATA
from src.config.schema.chosen_columns import CHOSEN_COLUMNS
from src.config.schema.columns_schema import COLUMNS_SCHEMA
from src.config.data.columns import GDELT_EVENT_COLUMNS
from src.config.data.column_type import (
    INTEGER_COLUMNS,
    FLOAT_COLUMNS,
    STRING_COLUMNS,
)
from src.utils.constants import NULL_TOKEN
from src.utils.loading import load_parquet
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)



# ---------------------------------------------------------------------
# Missing value handling (unchanged, but scoped properly)
# ---------------------------------------------------------------------
class MissingValueHandler:
    """Apply per-column missing-value policies declared in COLUMNS_SCHEMA."""

    def __init__(self, columns_schema: dict):
        self.columns_schema = columns_schema

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply per-column missing-value policies and return the modified DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing columns declared in columns_schema.

        Returns
        -------
        pd.DataFrame
            Copy of df with missing values handled according to each column's policy.
        """
        df = df.copy()

        for col, meta in self.columns_schema.items():
            if col not in df.columns:
                continue

            policy = meta.get("missing")

            if policy == "explicit":
                df[col] = df[col].fillna(NULL_TOKEN).astype("string")
                logger.debug(f"filling missing values in {col} with explicit {NULL_TOKEN}")

            elif policy == "indicator":
                indicator_col = f"{col}__is_missing"
                df[indicator_col] = df[col].isna().astype("int8")
                logger.debug(f"filling missing values in {col} with indicator {indicator_col}")

            elif policy == "error":
                if df[col].isna().any():
                    raise ValueError(
                        f"Missing values found in non-nullable column '{col}'"
                    )

        return df


# ---------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------
class DataCleaner:
    """
    Codebook-faithful data cleaning for GDELT event data.

    Responsibilities:
    - Validate columns against GDELT_EVENT_COLUMNS
    - Select a column subset (optional)
    - Cast columns to correct dtypes
    - Apply missing-value policies

    This class is:
    - Stateless
    - Split-agnostic
    - Model-agnostic
    """

    def __init__(
        self,
        sample_name: str,
        columns: Optional[Iterable[str]] = None,
    ):
        """
        Parameters
        ----------
        sample_name : str
            Name of the dataset sample directory.
        columns : Optional[Iterable[str]]
            Columns to retain. Must be a subset of GDELT_EVENT_COLUMNS.
            If None, defaults to CHOSEN_COLUMNS.
        """
        self.sample_name = sample_name
        self.input_dir = RAW_DATA 
        self.output_dir = PROCESSED_DATA / sample_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.selected_columns = list(columns) if columns is not None else list(CHOSEN_COLUMNS)

        logger.info(
            f"""
            Initializing DataCleaner for sample {self.sample_name}
            Input dir: {self.input_dir}
            Output dir: {self.output_dir}
            Selected columns: {len(self.selected_columns)}
            """)
        
        self._validate_column_policy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_column_policy(self) -> None:
        """Raise if any requested column is not in the official GDELT event codebook."""
        unknown = set(self.selected_columns) - set(GDELT_EVENT_COLUMNS)
        if unknown:
            raise ValueError(
                f"Unknown columns requested (not in GDELT_EVENT_COLUMNS): {sorted(unknown)}"
            )

        missing_from_schema = set(GDELT_EVENT_COLUMNS) - set(self.selected_columns)
        if missing_from_schema:
            warnings.warn(
                f"{len(missing_from_schema)} GDELT columns are excluded "
                f"by the column policy.",
                UserWarning,
            )
        logger.info(
            f"{len(missing_from_schema)} GDELT columns excluded by column policy"
            f"Excluded columns: {sorted(missing_from_schema)}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_raw(self, filename: str) -> pd.DataFrame:
        """Load a raw parquet file from the configured input directory."""
        logger.info(f"Loading raw data file '{filename}' from {self.input_dir}")
        df = load_parquet(filename, self.input_dir)
        logger.info(f"Loaded dataframe with shape {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------
    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df containing only the configured selected columns."""
        missing = set(self.selected_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"Input data is missing required columns: {sorted(missing)}"
            )
        logger.debug(f"Selecting {len(self.selected_columns)} columns")
        return df[self.selected_columns].copy()

    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast each column to its canonical dtype (Int64, float32, or string)."""
        df = df.copy()

        for col in INTEGER_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("Int64")
                logger.debug(f"Casting {col} to Int64")

        for col in FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("float32")
                logger.debug(f"Casting {col} to float32")

        for col in STRING_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("string")
                logger.debug(f"Casting {col} to string")

        return df
    
    def normalize_targets(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """
        Ensure classification targets are zero-based contiguous integers.
        """
        df = df.copy()

        for col in target_cols:
            if col not in df.columns:
                continue

            if not pd.api.types.is_integer_dtype(df[col]):
                raise TypeError(f"Target column '{col}' must be integer-typed")

            min_val = df[col].min()
            max_val = df[col].max()

            # GDELT case: QuadClass labels are 1-based (1 to 4)
            if min_val == 1:
                logger.info(f"Normalizing target '{col}' from 1-based to 0-based indexing")
                df[col] = df[col] - 1
                min_val -= 1
                max_val -= 1

            # Final safety check
            expected = set(range(int(max_val) + 1))
            actual = set(df[col].unique())

            if actual != expected:
                raise ValueError(
                    f"Target '{col}' is not contiguous after normalization. "
                    f"Expected {expected}, got {sorted(actual)}"
                )

        return df

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save(self, df: pd.DataFrame, filename: str) -> Path:
        """Write df as processed_{filename}.parquet in the output directory; return the path."""
        path = self.output_dir / f"processed_{filename}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Saving processed data to {path}")
        return path

    # ------------------------------------------------------------------
    # High-level pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        sample_name: str,
        target_cols: Optional[List[str]] = None,
    ) -> Path:
        """
        Execute the full cleaning pipeline and return the output path.

        Steps: load raw -> select columns -> cast types ->
        drop/normalise targets (optional) -> handle missing values -> save.

        Parameters
        ----------
        sample_name : str
            Filename (without extension) of the raw parquet to process.
        target_cols : list[str] or None
            Columns treated as classification targets. Rows with missing
            target values are dropped and labels are shifted to be
            zero-based before saving.

        Returns
        -------
        Path
            Path to the written processed parquet file.
        """
        logger.info(f"Running data cleaning pipeline for '{sample_name}'")
        df = self.load_raw(sample_name)

        df = self.select_columns(df)
        df = self.cast_types(df)

        if target_cols:
            n = len(df)
            df = df.dropna(subset=target_cols)
            logger.info(f"Dropped {n-len(df)} rows due to missing target columns {target_cols}")
            
            df = self.normalize_targets(df, target_cols)

        df = MissingValueHandler(COLUMNS_SCHEMA).apply(df)

        return self.save(df, sample_name)
