# src/utils/class_weights.py
import numpy as np
import torch 

def compute_class_weights(y: np.ndarray, num_classes: int | None = None) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Classes absent from y receive weight 0 so they do not affect the loss
    even when num_classes is larger than the number of observed classes.
    The returned weights are normalised so that the mean weight over
    non-empty classes equals 1.0.

    Parameters
    ----------
    y : np.ndarray of shape (N,)
        Integer class labels.
    num_classes : int or None
        Total number of classes. When None, inferred from y.

    Returns
    -------
    torch.Tensor of shape (num_classes,)
        Float32 inverse-frequency weights, one per class.
    """
    counts = np.bincount(y, minlength=num_classes or 0)
    counts = counts.astype(np.float32)

    # Avoid division by zero
    nonzero = counts > 0
    weights = np.zeros_like(counts, dtype=np.float32)

    weights[nonzero] = 1.0 / counts[nonzero]

    # Normalize so mean weight = 1
    weights = weights / weights[nonzero].mean()

    return torch.tensor(weights, dtype=torch.float32)
