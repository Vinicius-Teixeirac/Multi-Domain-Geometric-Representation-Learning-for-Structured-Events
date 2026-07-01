"""Tests for EventMLP: construction, forward pass, and forward_batch unpacking."""

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
def mlp() -> EventMLP:
    """Small EventMLP instance for fast unit tests."""
    return EventMLP(
        categorical_cardinalities=CATEGORICAL_CARDINALITIES,
        numeric_dim=NUMERIC_DIM,
        hidden_dims=[32, 16],
        num_classes=NUM_CLASSES,
        dropout=0.1,
    )


class TestEventMLP:
    def test_construction(self, mlp):
        """Model is a ``torch.nn.Module`` with the correct output width."""
        assert isinstance(mlp, torch.nn.Module)
        assert mlp.classifier.out_features == NUM_CLASSES

    def test_forward_shape(self, mlp, sample_x_cat, sample_x_num):
        """Forward pass returns float32 logits of shape ``(BATCH_SIZE, NUM_CLASSES)``."""
        mlp.eval()
        logits = mlp(sample_x_cat, sample_x_num)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert logits.dtype == torch.float32

    def test_forward_batch_unpacking(self, mlp, sample_x_cat, sample_x_num):
        """
        ``forward_batch`` unpacks a ``(x_cat, x_num, targets)`` tuple and
        returns logits and targets without modifying the target tensor.
        """
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        batch = (sample_x_cat, sample_x_num, targets)

        mlp.eval()
        logits, tgts = mlp.forward_batch(batch, "cpu")

        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert tgts.shape == (BATCH_SIZE,)
        assert tgts.dtype == torch.long
        assert torch.equal(tgts, targets)
