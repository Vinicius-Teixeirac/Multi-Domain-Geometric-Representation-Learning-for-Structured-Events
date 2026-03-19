from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

@torch.no_grad()
def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y = targets.cpu().numpy()

    return {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        # "f1_micro": f1_score(y, preds, average="micro"),
        "f1_weighted": f1_score(y, preds, average="weighted"),
        "precision_macro": precision_score(y, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y, preds, average="macro", zero_division=0),
    }

@torch.no_grad()
def compute_confusion(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> np.ndarray:
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y = targets.cpu().numpy()
    return confusion_matrix(y, preds)