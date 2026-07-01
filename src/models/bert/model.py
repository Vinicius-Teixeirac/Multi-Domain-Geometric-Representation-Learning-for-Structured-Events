# src/models/bert/model.py
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class BertForQuadClass(nn.Module):
    """
    BERT-based text classifier for the 4-class CAMEO QuadClass prediction task.

    The final hidden state of the [CLS] token is passed through a linear
    classifier.  Early transformer layers can be frozen to reduce the number
    of trainable parameters and stabilise fine-tuning on smaller datasets.

    Parameters
    ----------
    num_classes : int
        Number of output classes (typically 4 for QuadClass).
    model_name : str
        HuggingFace model identifier (e.g. 'bert-base-uncased').
    freeze_until_layer : int
        All transformer layers with index < freeze_until_layer will have
        their parameters frozen during training.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "bert-base-uncased",
        freeze_until_layer: int = 10,
    ):
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes.
        model_name : str
            HuggingFace model identifier.
        freeze_until_layer : int
            Layers below this index are frozen.
        """
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze early layers
        for name, param in self.bert.named_parameters():
            layer_num = self._get_layer_num(name)
            if layer_num is not None and layer_num < freeze_until_layer:
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run BERT and return class logits from the [CLS] token.

        Parameters
        ----------
        input_ids : torch.Tensor of shape (B, seq_len)
        attention_mask : torch.Tensor of shape (B, seq_len)

        Returns
        -------
        torch.Tensor of shape (B, num_classes)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_embedding)

    def forward_batch(
        self,
        batch: dict,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack a BertDataset batch, move tensors to device, and return (logits, targets).

        Parameters
        ----------
        batch : dict with keys 'input_ids', 'attention_mask', 'labels'
        device : str
            Target device string.

        Returns
        -------
        tuple of (logits, targets) both on device.
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)
        logits = self.forward(input_ids, attention_mask)
        return logits, targets

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    @staticmethod
    def _get_layer_num(param_name: str) -> Optional[int]:
        """
        Extract layer number from parameter name.
        Returns None if not a transformer layer.
        """
        if param_name.startswith("encoder.layer."):
            try:
                return int(param_name.split(".")[2])
            except Exception:
                return None
        return None
