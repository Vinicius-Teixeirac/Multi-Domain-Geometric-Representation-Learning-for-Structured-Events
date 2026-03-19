# src/models/mlp/datamodule.py
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from torch.utils.data import DataLoader

from src.config.paths import FEATURES_DATA, ARTIFACTS_DATA
from src.config.schema.columns_schema import COLUMNS_SCHEMA
from src.config.schema.encoding_schema import ENCODING_SCHEMA
from src.representation.tabular.encoding import SafeLabelEncoder, HashEncoder
from src.utils.loading import load_parquet

from .dataset import EventDataset


class EventDataModule:
    """
    DataModule built on top of TabularPipeline outputs.

    Responsibilities:
    - Load encoded tabular features
    - Infer categorical vs numeric feature groups using schema
    - Load categorical cardinalities for embedding layers
    - Provide PyTorch DataLoaders
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        dataset_name: str,
        split_tag: str = "default",
        batch_size: int = 512,
        num_workers: int = 0,
        target_col: str = "QuadClass",
    ):
        self.dataset_name = dataset_name
        self.split_tag = split_tag
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_col = target_col

        self.data_dir = FEATURES_DATA / dataset_name
        self.artifacts_dir = ARTIFACTS_DATA / dataset_name / "features" / split_tag

        self.train_df: Optional[pd.DataFrame] = None
        self.valid_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cardinalities: Dict[str, int] = {}
        self.numeric_dim: int = 0
        self.num_classes: Optional[int] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self) -> None:
        # --- load encoded feature splits ---
        self.train_df = load_parquet(
            f"train_{self.split_tag}_features.parquet",
            self.data_dir,
        )

        try:
            self.valid_df = load_parquet(
                f"valid_{self.split_tag}_features.parquet",
                self.data_dir,
            )
        except FileNotFoundError:
            self.valid_df = None

        self.test_df = load_parquet(
            f"test_{self.split_tag}_features.parquet",
            self.data_dir,
        )

        # --- infer feature groups ---
        self._infer_feature_groups()
        self._infer_num_classes()

    # ------------------------------------------------------------------
    # Feature inference
    # ------------------------------------------------------------------
    def _infer_feature_groups(self) -> None:
        """
        Infer categorical and numeric feature groups using COLUMNS_SCHEMA.

        Rules:
        - target column is excluded
        - id columns are defensively excluded
        - categorical columns follow schema
        - numeric columns = everything else
        """

        for col in self.train_df.columns:
            if col == self.target_col:
                continue

            schema = COLUMNS_SCHEMA.get(col)

            # Defensive: drop ID columns if they survive preprocessing
            if schema is not None and schema["kind"] == "id":
                continue

            if schema is not None and schema["kind"] == "categorical":
                self.categorical_cols.append(col)
            else:
                self.numeric_cols.append(col)

        self.categorical_cardinalities = self._load_categorical_cardinalities()
        self.numeric_dim = len(self.numeric_cols)

    def _infer_num_classes(self) -> None:
        """
        Infer number of target classes from training data.
        Assumes target is integer-encoded and zero- or arbitrary-based.
        """
        y = self.train_df[self.target_col].to_numpy()
        self.num_classes = int(pd.unique(y).size)


    # ------------------------------------------------------------------
    # Cardinalities
    # ------------------------------------------------------------------
    def _load_categorical_cardinalities(self) -> Dict[str, int]:
        """
        Load embedding cardinalities from encoder artifacts.
        """
        cardinalities: Dict[str, int] = {}

        for col in self.categorical_cols:
            enc_cfg = ENCODING_SCHEMA.get(col)
            if enc_cfg is None:
                continue

            path = self.artifacts_dir / f"{col}.json"
            if not path.exists():
                continue

            method = enc_cfg["method"]

            if method == "label":
                enc = SafeLabelEncoder.load(path)
                cardinalities[col] = enc.num_classes_
            elif method == "hash":
                enc = HashEncoder.load(path)
                cardinalities[col] = enc.num_buckets
            else:
                continue

            # cardinalities[col] = enc.cardinality

        return cardinalities

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def _make_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = EventDataset(
            dataframe=df,
            categorical_cols=self.categorical_cols,
            numeric_cols=self.numeric_cols,
            target_col=self.target_col,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_df, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.valid_df is None:
            return None
        return self._make_loader(self.valid_df, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_df, shuffle=False)
