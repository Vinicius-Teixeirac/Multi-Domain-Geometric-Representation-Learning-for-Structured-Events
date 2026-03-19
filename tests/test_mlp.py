import torch
import pytest

from src.models.mlp.model import EventMLP
from tests.conftest import (
    CATEGORICAL_CARDINALITIES,
    NUMERIC_DIM,
    NUM_CLASSES,
    BATCH_SIZE,
)


@pytest.fixture
def mlp():
    return EventMLP(
        categorical_cardinalities=CATEGORICAL_CARDINALITIES,
        numeric_dim=NUMERIC_DIM,
        hidden_dims=[32, 16],
        num_classes=NUM_CLASSES,
        dropout=0.1,
    )


class TestEventMLP:
    def test_construction(self, mlp):
        assert isinstance(mlp, torch.nn.Module)
        assert mlp.classifier.out_features == NUM_CLASSES

    def test_forward_shape(self, mlp, sample_x_cat, sample_x_num):
        mlp.eval()
        logits = mlp(sample_x_cat, sample_x_num)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert logits.dtype == torch.float32

    def test_forward_batch_unpacking(self, mlp, sample_x_cat, sample_x_num):
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        batch = (sample_x_cat, sample_x_num, targets)

        mlp.eval()
        logits, tgts = mlp.forward_batch(batch, "cpu")

        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert tgts.shape == (BATCH_SIZE,)
        assert tgts.dtype == torch.long
        assert torch.equal(tgts, targets)
