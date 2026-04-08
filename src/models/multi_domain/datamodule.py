# src/models/multi_domain/datamodule.py
"""
MultiDomainDataModule

Orchestrates data loading and pre-processing for the MultiDomainGeometricModel:

  1. Load cleaned split parquets from SPLITS_DATA (raw columns).
  2. Build the actor co-occurrence graph (train edges only, all-split actor nodes).
  3. Build country → index mapping for the geo encoder (region_aware type).
  4. Wrap each split in a MultiDomainEventDataset.
  5. Expose DataLoaders, the actor graph, and cardinalities for model initialisation.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from src.config.paths import SPLITS_DATA
from src.utils.loading import load_parquet
from src.representation.multi_domain.actor_graph_builder import build_actor_graph

from .dataset import MultiDomainEventDataset

_GEO_COUNTRY_COL = "ActionGeo_CountryCode"


class MultiDomainDataModule:
    """
    DataModule for the multi-domain geometric event model.

    Parameters
    ----------
    dataset_name : dataset directory name (e.g. 'sample_500000')
    split_tag : split identifier (e.g. 'default_s42')
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

        self.actor_graph           = None
        self.actor_to_idx:         dict[str, int] = {}
        self.actor_cardinalities:  list[int] = []

        self.country_to_idx:          dict[str, int] = {}
        self.geo_country_cardinality: int = 1  # safe default (1 = unknown only)

        self.train_dataset: Optional[MultiDomainEventDataset] = None
        self.valid_dataset: Optional[MultiDomainEventDataset] = None
        self.test_dataset:  Optional[MultiDomainEventDataset] = None

        self.num_classes: Optional[int] = None

    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Load data, build actor graph and country index, create datasets."""

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

        # 2. Build actor co-occurrence graph
        self.actor_graph, self.actor_to_idx, self.actor_cardinalities, _ = \
            build_actor_graph(self.train_df)

        # 3. Build country → index mapping (index 0 = unknown/missing)
        all_dfs = (
            [self.train_df]
            + ([] if self.valid_df is None else [self.valid_df])
            + [self.test_df]
        )
        country_series = [
            df[_GEO_COUNTRY_COL].fillna("").astype(str)
            for df in all_dfs
            if _GEO_COUNTRY_COL in df.columns
        ]
        if country_series:
            unique_countries = sorted(
                c for c in pd.concat(country_series).unique() if c != ""
            )
            self.country_to_idx = {c: i + 1 for i, c in enumerate(unique_countries)}
        else:
            self.country_to_idx = {}
        self.geo_country_cardinality = len(self.country_to_idx) + 1  # +1 for unknown

        # 4. Create datasets (fit temporal stats on train, apply to val/test)
        self.train_dataset = MultiDomainEventDataset(
            df=self.train_df,
            actor_to_idx=self.actor_to_idx,
            country_to_idx=self.country_to_idx,
        )
        time_mean = self.train_dataset.time_mean
        time_std  = self.train_dataset.time_std

        if self.valid_df is not None:
            self.valid_dataset = MultiDomainEventDataset(
                df=self.valid_df,
                actor_to_idx=self.actor_to_idx,
                country_to_idx=self.country_to_idx,
                time_mean=time_mean,
                time_std=time_std,
            )

        self.test_dataset = MultiDomainEventDataset(
            df=self.test_df,
            actor_to_idx=self.actor_to_idx,
            country_to_idx=self.country_to_idx,
            time_mean=time_mean,
            time_std=time_std,
        )

    # ------------------------------------------------------------------
    def _make_loader(self, dataset: MultiDomainEventDataset, shuffle: bool) -> DataLoader:
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
