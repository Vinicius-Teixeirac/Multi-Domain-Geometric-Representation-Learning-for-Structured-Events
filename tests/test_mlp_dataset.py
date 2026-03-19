import torch
import pytest

from src.models.mlp.dataset import EventDataset
from tests.conftest import NUMERIC_DIM, NUM_CLASSES


class TestEventDataset:
    def test_len(self, sample_dataframe):
        ds = EventDataset(
            dataframe=sample_dataframe,
            categorical_cols=["col_a", "col_b"],
            numeric_cols=["num_0", "num_1", "num_2", "num_3"],
        )
        assert len(ds) == len(sample_dataframe)

    def test_getitem_format(self, sample_dataframe):
        ds = EventDataset(
            dataframe=sample_dataframe,
            categorical_cols=["col_a", "col_b"],
            numeric_cols=["num_0", "num_1", "num_2", "num_3"],
        )
        cat_dict, num_vec, target = ds[0]

        assert isinstance(cat_dict, dict)
        assert set(cat_dict.keys()) == {"col_a", "col_b"}
        for v in cat_dict.values():
            assert v.dtype == torch.long
            assert v.ndim == 0  # scalar

        assert num_vec.shape == (NUMERIC_DIM,)
        assert num_vec.dtype == torch.float32

        assert target.dtype == torch.long
        assert target.ndim == 0

    def test_no_numeric(self, sample_dataframe_no_numeric):
        ds = EventDataset(
            dataframe=sample_dataframe_no_numeric,
            categorical_cols=["col_a", "col_b"],
            numeric_cols=[],
        )
        cat_dict, num_vec, target = ds[0]
        assert num_vec.shape == (0,)
        assert num_vec.dtype == torch.float32
