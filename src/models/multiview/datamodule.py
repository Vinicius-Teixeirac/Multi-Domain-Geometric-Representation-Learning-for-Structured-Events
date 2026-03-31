# src/models/multiview/datamodule.py
"""
MultiviewDataModule

Orchestrates data loading and pre-processing for the MultiviewGeometricModel:

  1. Load cleaned split parquets from SPLITS_DATA (raw columns, not tabular-encoded).
  2. Build the actor co-occurrence graph (train edges only, all-split actor nodes).
  3. Compute actor → node-index mappings per split.
  4. Wrap each split in a MultiviewEventDataset.
  5. Expose DataLoaders and the actor graph for model initialisation.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from src.config.paths import SPLITS_DATA
from src.utils.loading import load_parquet
from src.representation.multiview.actor_graph_builder import build_actor_graph

from .dataset import MultiviewEventDataset


class MultiviewDataModule:
    """
    DataModule for the multiview geometric event model.

    Parameters
    ----------
    dataset_name : dataset directory name (e.g. 'sample_500000')
    split_tag : split identifier (e.g. 'default')
    batch_size : mini-batch size for DataLoaders
    num_workers : DataLoader worker processes
    target_col : target column name
    """

    def __init__(
        self,
        dataset_name: str,
        split_tag: str = "default",
        batch_size: int = 512,
        num_workers: int = 0,
        target_col: str = "QuadClass",
    ):
        self.dataset_name = dataset_name
        self.split_tag    = split_tag
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.target_col   = target_col

        self.splits_dir = SPLITS_DATA / dataset_name

        # Set after setup()
        self.train_df: Optional[pd.DataFrame] = None
        self.valid_df: Optional[pd.DataFrame] = None
        self.test_df:  Optional[pd.DataFrame] = None

        self.actor_graph      = None   # torch_geometric.data.Data
        self.actor_to_idx: dict[str, int] = {}
        self.actor_cardinalities: list[int] = []

        self.train_dataset: Optional[MultiviewEventDataset] = None
        self.valid_dataset: Optional[MultiviewEventDataset] = None
        self.test_dataset:  Optional[MultiviewEventDataset] = None

        self.num_classes: Optional[int] = None

    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Load data, build actor graph, create datasets."""

        # 1. Load splits
        self.train_df = load_parquet(
            f"train_{self.split_tag}.parquet", self.splits_dir
        )
        try:
            self.valid_df = load_parquet(
                f"valid_{self.split_tag}.parquet", self.splits_dir
            )
        except FileNotFoundError:
            self.valid_df = None

        self.test_df = load_parquet(
            f"test_{self.split_tag}.parquet", self.splits_dir
        )

        self.num_classes = int(self.train_df[self.target_col].nunique())

        # 2. Build actor graph (all split actor IDs as nodes, train edges only)
        self.actor_graph, self.actor_to_idx, self.actor_cardinalities, _ = \
            build_actor_graph(self.train_df, self.valid_df, self.test_df)

        # 3. Create datasets (fit temporal stats on train, apply to val/test)
        self.train_dataset = MultiviewEventDataset(
            df=self.train_df,
            actor_to_idx=self.actor_to_idx,
        )
        time_mean = self.train_dataset.time_mean
        time_std  = self.train_dataset.time_std

        if self.valid_df is not None:
            self.valid_dataset = MultiviewEventDataset(
                df=self.valid_df,
                actor_to_idx=self.actor_to_idx,
                time_mean=time_mean,
                time_std=time_std,
            )

        self.test_dataset = MultiviewEventDataset(
            df=self.test_df,
            actor_to_idx=self.actor_to_idx,
            time_mean=time_mean,
            time_std=time_std,
        )

    # ------------------------------------------------------------------
    def _make_loader(self, dataset: MultiviewEventDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.valid_dataset is None:
            return None
        return self._make_loader(self.valid_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)
