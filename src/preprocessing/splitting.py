"""
splitting.py

Responsible for splitting cleaned datasets into:
- train
- validation (optional)
- test

Splitting happens *after* cleaning and *before* any encoding.

NOTE:
This splitter enforces *event-level inductive* splits.
Entities (actors, locations, days) may appear across splits.
"""

from pathlib import Path
from typing import Optional, Literal
import warnings
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.paths import PROCESSED_DATA, SPLITS_DATA
from src.utils.loading import load_parquet
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)

SplitStrategy = Literal["random", "temporal"]
"""random: shuffled stratified split; temporal: chronological split (requires time_column)."""

# Minimum combined train+val size before we skip creating a separate validation set.
# Below this the validation set would be too small to give reliable metric estimates.
_MIN_TRAIN_FOR_VALID: int = 2000


class Splitter:
    """
    Event-level inductive dataset splitter for GDELT event data.

    Entities (actors, locations, days) may appear across splits; only
    event identity is strictly isolated between train, validation, and test.

    Parameters
    ----------
    sample_name : str
        Dataset directory name (must exist under PROCESSED_DATA).
    """

    def __init__(self, sample_name: str):
        """
        Parameters
        ----------
        sample_name : str
            Dataset directory name (must exist under PROCESSED_DATA).
        """
        self.sample_name = sample_name
        self.input_dir = PROCESSED_DATA / sample_name
        self.output_dir = SPLITS_DATA / sample_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"""
            Initializing Splitter for sample {self.sample_name}
            Input dir: {self.input_dir}
            Output dir: {self.output_dir}
            """
        )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_clean(self, filename: str) -> pd.DataFrame:
        """Load a cleaned parquet file from the input directory."""
        return load_parquet(filename, self.input_dir)

    # ------------------------------------------------------------------ #
    # Temporal ordering
    # ------------------------------------------------------------------ #
    @staticmethod
    def sort_by_time(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Sort df ascending by column (in-place chronological ordering for temporal splits)."""
        if column not in df.columns:
            raise ValueError(f"Time column '{column}' not found in dataframe.")
        return df.sort_values(column).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def _save(self, df: pd.DataFrame, split: str, tag: str) -> Path:
        """Write split parquet to the output directory; return the path."""
        filename = f"{split}_{tag}.parquet"
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        logger.debug(f"Saved {split} split to {path}")
        return path

    def _save_metadata(self, tag: str, metadata: dict) -> None:
        """Persist split configuration metadata as split_{tag}.json."""
        path = self.output_dir / f"split_{tag}.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved split metadata to {path}")

    # ------------------------------------------------------------------ #
    # Main logic
    # ------------------------------------------------------------------ #
    def run(
        self,
        filename: str,
        *,
        strategy: SplitStrategy = "random",
        test_size: float = 0.15,
        valid_size: Optional[float] = 0.10,
        stratify_by: Optional[str] = None,
        time_column: Optional[str] = None,
        random_state: int,
        tag: str = "default",
    ) -> None:
        """
        Perform dataset splitting and save results.

        Parameters
        ----------
        filename : str
            Cleaned parquet filename.
        strategy : {"random", "temporal"}
            Splitting strategy. Temporal implies chronological ordering.
        test_size : float
            Fraction of data used for test split.
        valid_size : float or None
            Fraction of full dataset used for validation.
            If None, no validation split is created.
        stratify_by : str or None
            Column name for stratified splitting (random only).
        time_column : str or None
            Column name for chronological splitting (temporal only).
        random_state : int
            Global experiment seed.
        tag : str
            Semantic identifier for this split regime.
        """
        logger.info(
            f"""
            Starting dataset split
            file={filename}
            strategy={strategy}
            test_size={test_size}
            valid_size={valid_size}
            tag={tag}
            """
        )

        df = self.load_clean(filename)

        # --------------------------------------------------------------
        # Strategy validation
        # --------------------------------------------------------------
        if strategy == "random":
            if time_column is not None:
                raise ValueError("time_column not allowed for random strategy")

        elif strategy == "temporal":
            if time_column is None:
                raise ValueError("Temporal strategy requires time_column")
            if stratify_by is not None:
                warnings.warn("Stratification ignored for temporal splits")
                logger.warning("Stratification ignored for temporal splits")
                stratify_by = None
            df = self.sort_by_time(df, time_column)

        else:
            raise ValueError(f"Unknown split strategy: {strategy}")

        # --------------------------------------------------------------
        # First split: train+val vs test
        # --------------------------------------------------------------
        strat_vec = df[stratify_by] if stratify_by else None

        df_train_valid, df_test = train_test_split(
            df,
            test_size=test_size,
            stratify=strat_vec,
            shuffle=(strategy == "random"),
            random_state=random_state,
        )

        logger.info(
            f"Split sizes - train+val: {len(df_train_valid)}, test: {len(df_test)}"
        )

        self._save(df_test, "test", tag)

        # --------------------------------------------------------------
        # Optional validation split
        # --------------------------------------------------------------
        if valid_size is None:
            logger.info("Validation split disabled; saving full train set")
            self._save(df_train_valid, "train", tag)
            self._save_metadata(
                tag,
                {
                    "strategy": strategy,
                    "test_size": test_size,
                    "valid_size": None,
                    "stratify_by": stratify_by,
                    "time_column": time_column,
                    "random_state": random_state,
                    "num_rows": len(df),
                },
            )
            return

        if len(df_train_valid) < _MIN_TRAIN_FOR_VALID:
            warnings.warn(
                f"Train set has only {len(df_train_valid)} instances (< {_MIN_TRAIN_FOR_VALID}); "
                "skipping validation split."
            )
            logger.warning(
                f"Skipping validation split due to small dataset size ({len(df_train_valid)})"
            )
            self._save(df_train_valid, "train", tag)
            self._save_metadata(
                tag,
                {
                    "strategy": strategy,
                    "test_size": test_size,
                    "valid_size": None,
                    "stratify_by": stratify_by,
                    "time_column": time_column,
                    "random_state": random_state,
                    "num_rows": len(df),
                },
            )
            return

        relative_val_size = valid_size / (1.0 - test_size)
        strat_vec_tv = (
            df_train_valid[stratify_by] if stratify_by else None
        )

        df_train, df_valid = train_test_split(
            df_train_valid,
            test_size=relative_val_size,
            stratify=strat_vec_tv,
            shuffle=(strategy == "random"),
            random_state=random_state,
        )

        logger.info(
            f"Final split sizes - train: {len(df_train)}, "
            f"valid: {len(df_valid)}, test: {len(df_test)}"
        )

        self._save(df_train, "train", tag)
        self._save(df_valid, "valid", tag)

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------
        self._save_metadata(
            tag,
            {
                "strategy": strategy,
                "test_size": test_size,
                "valid_size": valid_size,
                "stratify_by": stratify_by,
                "time_column": time_column,
                "random_state": random_state,
                "num_rows": len(df),
            },
        )
