# src/models/bert/datamodule.py

from pathlib import Path
from typing import Tuple

import pandas as pd
from torch.utils.data import DataLoader

from src.config.paths import TEXT_DATA
from src.representation.text.text_pipeline import TextPipeline
from src.models.bert.dataset import BertDataset


class BertEventDataModule:
    """
    DataModule for BERT-based text classification.

    Expects cached parquet files with columns:
        - text
        - label
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        batch_size: int,
        split_tag: str = "default",
        num_workers: int = 0,
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
    ):
        self.dataset_name = dataset_name
        self.split_tag = split_tag
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pipeline = TextPipeline(
            model_name=model_name,
            max_length=max_length,
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        train_df = self._load_split("train", self.split_tag)
        val_df = self._load_split("valid", self.split_tag)
        test_df = self._load_split("test", self.split_tag)

        self.train_dataset = self._build_dataset(train_df)
        self.val_dataset = self._build_dataset(val_df)
        self.test_dataset = self._build_dataset(test_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_path(self, split: str, split_tag: str) -> Path:
        return (
            TEXT_DATA
            / self.dataset_name
            / f"{split}_{split_tag}_text.parquet"
        )

    def _load_split(self, split: str, split_tag: str) -> pd.DataFrame:
        path = self._split_path(split, split_tag)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing text data for split='{split}', tag='{split_tag}': {path}"
            )
        return pd.read_parquet(path)

    def _build_dataset(self, df: pd.DataFrame) -> BertDataset:
        encodings, labels = self.pipeline.build_dataset(df)
        return BertDataset(encodings, labels)

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def _loader(self, dataset: BertDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
