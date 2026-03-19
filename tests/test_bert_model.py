import torch
import pytest

from src.models.bert.model import BertForQuadClass

NUM_CLASSES = 4


@pytest.fixture(scope="module")
def tiny_bert():
    return BertForQuadClass(
        num_classes=NUM_CLASSES,
        model_name="hf-internal-testing/tiny-bert",
        freeze_until_layer=0,
    )


class TestBertForQuadClass:
    def test_forward_batch_shape(self, tiny_bert):
        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "labels": torch.randint(0, NUM_CLASSES, (4,)),
        }
        tiny_bert.eval()
        logits, targets = tiny_bert.forward_batch(batch, "cpu")

        assert logits.shape == (4, NUM_CLASSES)
        assert logits.dtype == torch.float32
        assert targets.shape == (4,)
        assert targets.dtype == torch.long

    def test_forward_shape(self, tiny_bert):
        tiny_bert.eval()
        input_ids = torch.randint(0, 100, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        logits = tiny_bert.forward(input_ids, attention_mask)
        assert logits.shape == (4, NUM_CLASSES)
        assert logits.dtype == torch.float32

    def test_get_layer_num(self):
        assert BertForQuadClass._get_layer_num("encoder.layer.5.foo") == 5
        assert BertForQuadClass._get_layer_num("embeddings.word") is None
        assert BertForQuadClass._get_layer_num("encoder.layer.11.bar") == 11
