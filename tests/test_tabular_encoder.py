import torch
import pytest

from src.models.tabular_encoder import TabularInputEncoder
from tests.conftest import (
    CATEGORICAL_CARDINALITIES,
    NUMERIC_DIM,
    BATCH_SIZE,
    expected_output_dim,
)


class TestTabularInputEncoder:
    def test_construction(self, cardinalities):
        enc = TabularInputEncoder(
            categorical_cardinalities=cardinalities,
            numeric_dim=NUMERIC_DIM,
        )
        assert isinstance(enc, torch.nn.Module)
        assert len(enc.embeddings) == len(cardinalities)

    def test_output_dim_sqrt(self, cardinalities):
        enc = TabularInputEncoder(
            categorical_cardinalities=cardinalities,
            numeric_dim=NUMERIC_DIM,
            embedding_dim_rule="sqrt",
        )
        assert enc.output_dim == expected_output_dim(
            cardinalities, NUMERIC_DIM, "sqrt"
        )

    def test_output_dim_log(self, cardinalities):
        enc = TabularInputEncoder(
            categorical_cardinalities=cardinalities,
            numeric_dim=NUMERIC_DIM,
            embedding_dim_rule="log",
        )
        assert enc.output_dim == expected_output_dim(
            cardinalities, NUMERIC_DIM, "log"
        )

    def test_forward_shape(self, cardinalities, sample_x_cat, sample_x_num):
        enc = TabularInputEncoder(
            categorical_cardinalities=cardinalities,
            numeric_dim=NUMERIC_DIM,
        )
        out = enc(sample_x_cat, sample_x_num)
        assert out.shape == (BATCH_SIZE, enc.output_dim)
        assert out.dtype == torch.float32

    def test_no_numeric(self, cardinalities, sample_x_cat):
        enc = TabularInputEncoder(
            categorical_cardinalities=cardinalities,
            numeric_dim=0,
        )
        x_num = torch.empty(BATCH_SIZE, 0, dtype=torch.float32)
        out = enc(sample_x_cat, x_num)
        assert out.shape == (BATCH_SIZE, enc.output_dim)

    def test_dropout_mode(self, cardinalities, sample_x_cat, sample_x_num):
        enc = TabularInputEncoder(
            categorical_cardinalities=cardinalities,
            numeric_dim=NUMERIC_DIM,
            embedding_dropout=0.5,
        )
        enc.train()
        out_train = enc(sample_x_cat, sample_x_num)

        enc.eval()
        out_eval = enc(sample_x_cat, sample_x_num)

        # Eval should be deterministic
        out_eval2 = enc(sample_x_cat, sample_x_num)
        assert torch.allclose(out_eval, out_eval2)
