"""Tests for BertDataset: length, item format, and constructor validation."""

from typing import Dict

import torch
import pytest

from src.models.bert.dataset import BertDataset


@pytest.fixture
def bert_encodings() -> Dict[str, torch.Tensor]:
    """
    Minimal tokenizer-style encoding dict for 16 samples at sequence length 64.

    Returns
    -------
    dict[str, torch.Tensor]
        Keys ``input_ids`` (random token ids in BERT vocab range) and
        ``attention_mask`` (all ones).
    """
    n = 16
    return {
        "input_ids": torch.randint(0, 30522, (n, 64)),
        "attention_mask": torch.ones(n, 64, dtype=torch.long),
    }


@pytest.fixture
def bert_labels() -> torch.Tensor:
    """Random integer class labels for 16 samples (4 classes)."""
    return torch.randint(0, 4, (16,))


class TestBertDataset:
    def test_len(self, bert_encodings, bert_labels):
        """Dataset length matches the number of label entries."""
        ds = BertDataset(bert_encodings, bert_labels)
        assert len(ds) == 16

    def test_getitem_format(self, bert_encodings, bert_labels):
        """
        Single-item retrieval returns a dict with input_ids, attention_mask,
        and a scalar long label under the ``labels`` key.
        """
        ds = BertDataset(bert_encodings, bert_labels)
        item = ds[0]

        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

        assert item["input_ids"].shape == (64,)
        assert item["labels"].ndim == 0
        assert item["labels"].dtype == torch.long

    def test_bad_encodings_type(self, bert_labels):
        """Passing a non-dict as encodings raises TypeError."""
        with pytest.raises(TypeError, match="encodings must be a dict"):
            BertDataset("not_a_dict", bert_labels)

    def test_bad_labels_type(self, bert_encodings):
        """Passing a plain list as labels raises TypeError."""
        with pytest.raises(TypeError, match="labels must be a torch.Tensor"):
            BertDataset(bert_encodings, [1, 2, 3])

    def test_shape_mismatch(self, bert_encodings):
        """Mismatched encoding rows vs. label length raises ValueError."""
        bad_labels = torch.randint(0, 4, (5,))
        with pytest.raises(ValueError, match="rows"):
            BertDataset(bert_encodings, bad_labels)
