import torch
import pytest

from src.models.bert.model import BertForQuadClass

NUM_CLASSES = 4


@pytest.fixture(scope="module")
def tiny_bert():
    """
    Module-scoped BertForQuadClass instance backed by a tiny HuggingFace model.

    Uses ``hf-internal-testing/tiny-bert`` to keep the test suite fast while
    still exercising the real BERT forward pass and classification head.
    All layers are unfrozen (``freeze_until_layer=0``).

    Returns
    -------
    BertForQuadClass
        Ready-to-use model in its default (training) state.
    """
    return BertForQuadClass(
        num_classes=NUM_CLASSES,
        model_name="hf-internal-testing/tiny-bert",
        freeze_until_layer=0,
    )


class TestBertForQuadClass:
    def test_forward_batch_shape(self, tiny_bert):
        """
        ``forward_batch`` returns logits of shape ``(B, num_classes)`` and
        long targets of shape ``(B,)`` extracted from the batch dict.
        """
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
        """``forward`` returns float32 logits of shape ``(B, num_classes)``."""
        tiny_bert.eval()
        input_ids = torch.randint(0, 100, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        logits = tiny_bert.forward(input_ids, attention_mask)
        assert logits.shape == (4, NUM_CLASSES)
        assert logits.dtype == torch.float32

    def test_get_layer_num(self):
        """
        ``_get_layer_num`` extracts the integer layer index from a parameter
        name, returning ``None`` for names that don't match the encoder pattern.
        """
        assert BertForQuadClass._get_layer_num("encoder.layer.5.foo") == 5
        assert BertForQuadClass._get_layer_num("embeddings.word") is None
        assert BertForQuadClass._get_layer_num("encoder.layer.11.bar") == 11
