# src/models/mlp/model.py
import torch
import torch.nn as nn

from src.models.tabular_encoder import TabularInputEncoder


class EventMLP(nn.Module):
    def __init__(
        self,
        categorical_cardinalities,
        numeric_dim,
        hidden_dims=[256, 128],
        num_classes = None,
        dropout=0.2,
    ):
        super().__init__()

        self.input_encoder = TabularInputEncoder(
            categorical_cardinalities=categorical_cardinalities,
            numeric_dim=numeric_dim,
        )

        layers = []
        prev_dim = self.input_encoder.output_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def forward(self, x_cat, x_num):
        x = self.input_encoder(x_cat, x_num)
        x = self.feature_extractor(x)
        return self.classifier(x)

    def forward_batch(self, batch, device):
        x_cat = {k: v.to(device) for k, v in batch[0].items()}
        x_num = batch[1].to(device)
        y = batch[2].to(device)

        logits = self(x_cat, x_num)
        return logits, y
