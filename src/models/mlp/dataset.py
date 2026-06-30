# src/models/mlp/dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class EventDataset(Dataset):
    """
    Dataset returning:
        (
            categorical_dict: Dict[str, LongTensor],
            numeric_tensor: FloatTensor,
            target: LongTensor
        )
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        dataframe: pd.DataFrame,
        categorical_cols: List[str],
        numeric_cols: List[str],
        target_col: str = "QuadClass",
    ):
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols

        # --- categorical features ---
        self.x_cat: Dict[str, torch.Tensor] = {
            col: torch.tensor(
                dataframe[col].to_numpy(dtype=np.int64),
                dtype=torch.long,
            )
            for col in self.categorical_cols
        }

        # --- numeric features ---
        if self.numeric_cols:
            self.x_num = torch.tensor(
                dataframe[self.numeric_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
        else:
            # Handle edge case: no numeric features
            self.x_num = torch.empty((len(dataframe), 0), dtype=torch.float32)

        # --- target ---
        self.y = torch.tensor(
            dataframe[self.target_col].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )

    # ------------------------------------------------------------------
    # PyTorch API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of events in the dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        """Return (x_cat_dict, x_num_tensor, label) for the event at position idx."""
        return (
            {col: tensor[idx] for col, tensor in self.x_cat.items()},
            self.x_num[idx],
            self.y[idx],
        )
