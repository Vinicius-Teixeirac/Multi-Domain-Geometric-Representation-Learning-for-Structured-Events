import torch
import pytest

from src.models.bert.dataset import BertDataset


@pytest.fixture
def bert_encodings():
    n = 16
    return {
        "input_ids": torch.randint(0, 30522, (n, 64)),
        "attention_mask": torch.ones(n, 64, dtype=torch.long),
    }


@pytest.fixture
def bert_labels():
    return torch.randint(0, 4, (16,))


class TestBertDataset:
    def test_len(self, bert_encodings, bert_labels):
        ds = BertDataset(bert_encodings, bert_labels)
        assert len(ds) == 16

    def test_getitem_format(self, bert_encodings, bert_labels):
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
        with pytest.raises(TypeError, match="encodings must be a dict"):
            BertDataset("not_a_dict", bert_labels)

    def test_bad_labels_type(self, bert_encodings):
        with pytest.raises(TypeError, match="labels must be a torch.Tensor"):
            BertDataset(bert_encodings, [1, 2, 3])

    def test_shape_mismatch(self, bert_encodings):
        bad_labels = torch.randint(0, 4, (5,))
        with pytest.raises(ValueError, match="rows"):
            BertDataset(bert_encodings, bad_labels)
