# src/models/bert/model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class BertForQuadClass(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "bert-base-uncased",
        freeze_until_layer: int = 10,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze early layers
        for name, param in self.bert.named_parameters():
            layer_num = self._get_layer_num(name)
            if layer_num is not None and layer_num < freeze_until_layer:
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_embedding)

    def forward_batch(self, batch, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)
        logits = self.forward(input_ids, attention_mask)
        return logits, targets

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    @staticmethod
    def _get_layer_num(param_name: str):
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
