"""Tests for classification metric computation shared by every model runner."""

import numpy as np
import torch
import pytest

from src.utils.metrics import compute_classification_metrics, compute_confusion

NUM_CLASSES = 4


@pytest.fixture
def perfect_predictions():
    """Logits that put all probability mass on the true class -> perfect metrics."""
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    logits = torch.nn.functional.one_hot(targets, NUM_CLASSES).float() * 10.0
    return logits, targets


class TestComputeClassificationMetrics:
    def test_perfect_predictions_score_1(self, perfect_predictions):
        """When logits exactly match targets, every metric should be 1.0."""
        logits, targets = perfect_predictions
        metrics = compute_classification_metrics(logits, targets)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["f1_weighted"] == pytest.approx(1.0)
        assert metrics["precision_macro"] == pytest.approx(1.0)
        assert metrics["recall_macro"] == pytest.approx(1.0)

    def test_returns_expected_keys(self, perfect_predictions):
        """The dict must expose exactly the metrics consumed downstream by train.py/evaluate.py."""
        logits, targets = perfect_predictions
        metrics = compute_classification_metrics(logits, targets)
        assert set(metrics.keys()) == {
            "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
        }

    def test_all_wrong_predictions_score_below_perfect(self):
        """Systematically wrong predictions must score strictly below the perfect case."""
        targets = torch.tensor([0, 1, 2, 3])
        wrong_class = (targets + 1) % NUM_CLASSES
        logits = torch.nn.functional.one_hot(wrong_class, NUM_CLASSES).float() * 10.0
        metrics = compute_classification_metrics(logits, targets)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_missing_class_does_not_crash_with_zero_division(self):
        """precision/recall must use zero_division=0 so an unpredicted class doesn't raise/warn-crash."""
        targets = torch.tensor([0, 0, 0, 0])
        logits = torch.nn.functional.one_hot(targets, NUM_CLASSES).float() * 10.0
        metrics = compute_classification_metrics(logits, targets)
        assert not any(np.isnan(v) for v in metrics.values())


class TestComputeConfusion:
    def test_shape_and_diagonal_for_perfect_predictions(self, perfect_predictions):
        """A perfect classifier produces a diagonal (num_classes, num_classes) confusion matrix."""
        logits, targets = perfect_predictions
        cm = compute_confusion(logits, targets)
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
        assert np.array_equal(cm, np.diag(np.diag(cm)))
        assert cm.sum() == len(targets)

    def test_off_diagonal_for_systematic_error(self):
        """Consistently predicting class (true+1) puts all mass one column to the right of the diagonal."""
        targets = torch.tensor([0, 1, 2])
        wrong_class = (targets + 1) % NUM_CLASSES
        logits = torch.nn.functional.one_hot(wrong_class, NUM_CLASSES).float() * 10.0
        cm = compute_confusion(logits, targets)
        for true_cls, pred_cls in zip(targets.tolist(), wrong_class.tolist()):
            assert cm[true_cls, pred_cls] >= 1
