"""Torch Dataset wrapping tokenized BERT encodings and labels."""

from typing import Dict
import torch


class BertDataset(torch.utils.data.Dataset):
    """
    Torch Dataset for BERT-based text classification.

    Parameters
    ----------
    encodings : Dict[str, Tensor]
        Tokenizer outputs (input_ids, attention_mask, ...)
    labels : Tensor
        Class labels (shape: [N])
    """

    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ):
        if not isinstance(encodings, dict):
            raise TypeError("encodings must be a dict of tensors")

        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor")

        self.encodings = encodings
        self.labels = labels

        self._validate_shapes()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single example as a dict of tensors including 'labels'."""
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate_shapes(self):
        """Raise ValueError if any encoding tensor has a different first-dimension size from labels."""
        n = self.labels.size(0)
        for key, tensor in self.encodings.items():
            if tensor.size(0) != n:
                raise ValueError(
                    f"Encoding '{key}' has {tensor.size(0)} rows, "
                    f"but labels have {n}"
                )
