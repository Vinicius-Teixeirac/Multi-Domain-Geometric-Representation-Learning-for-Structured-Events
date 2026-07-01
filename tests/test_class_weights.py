"""Tests for inverse-frequency class weighting."""

import numpy as np
import torch
import pytest

from src.utils.class_weights import compute_class_weights


class TestComputeClassWeights:
    def test_balanced_classes_get_equal_weight(self):
        """Equal class counts should all receive the same (normalised-to-1) weight."""
        y = np.array([0, 0, 1, 1, 2, 2])
        weights = compute_class_weights(y, num_classes=3)
        assert torch.allclose(weights, torch.ones(3), atol=1e-5)

    def test_rare_class_gets_higher_weight(self):
        """A class with fewer observations must receive a strictly larger weight."""
        y = np.array([0] * 90 + [1] * 10)
        weights = compute_class_weights(y, num_classes=2)
        assert weights[1] > weights[0]

    def test_absent_class_gets_zero_weight(self):
        """Classes never observed in y must get weight 0, not divide-by-zero garbage."""
        y = np.array([0, 0, 1, 1])
        weights = compute_class_weights(y, num_classes=4)
        assert weights[2] == 0.0
        assert weights[3] == 0.0
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()

    def test_mean_weight_over_nonzero_classes_is_one(self):
        """Weights are normalised so non-empty classes average to 1.0 (documented behavior)."""
        y = np.array([0, 0, 0, 1, 2, 2, 2, 2, 2])
        weights = compute_class_weights(y, num_classes=3)
        assert weights.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_num_classes_inferred_when_none(self):
        """num_classes=None should infer the class count from the observed labels."""
        y = np.array([0, 1, 2, 2])
        weights = compute_class_weights(y, num_classes=None)
        assert weights.shape == (3,)

    def test_output_dtype_is_float32(self):
        """Weights must be float32 so they plug directly into CrossEntropyLoss(weight=...)."""
        y = np.array([0, 1])
        weights = compute_class_weights(y, num_classes=2)
        assert weights.dtype == torch.float32
