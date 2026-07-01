"""Classification metric computation shared by all model runners' evaluation step."""

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
    """
    Compute standard multi-class classification metrics from raw logits.

    Parameters
    ----------
    logits : torch.Tensor of shape (N, C)
        Unnormalised class scores.
    targets : torch.Tensor of shape (N,)
        Ground-truth class indices.

    Returns
    -------
    dict with keys:
        accuracy         : float
        f1_macro         : float
        f1_weighted      : float
        precision_macro  : float
        recall_macro     : float
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y = targets.cpu().numpy()

    return {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        # f1_micro is mathematically identical to accuracy in the single-label
        # multi-class setting, so it's kept here (disabled) only as a reference
        # for readers unfamiliar with that equivalence, not as a metric gap.
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
    """
    Compute a confusion matrix from raw logits.

    Parameters
    ----------
    logits : torch.Tensor of shape (N, C)
        Unnormalised class scores.
    targets : torch.Tensor of shape (N,)
        Ground-truth class indices.

    Returns
    -------
    np.ndarray of shape (C, C)
        Confusion matrix where entry [i, j] is the number of samples
        with true class i predicted as class j.
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y = targets.cpu().numpy()
    return confusion_matrix(y, preds)