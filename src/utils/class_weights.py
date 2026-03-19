# src/utils/class_weights.py
import numpy as np
import torch 

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for CrossEntropyLoss.
    Safe against missing classes.
    """
    counts = np.bincount(y)
    counts = counts.astype(np.float32)

    # Avoid division by zero
    nonzero = counts > 0
    weights = np.zeros_like(counts, dtype=np.float32)

    weights[nonzero] = 1.0 / counts[nonzero]

    # Normalize so mean weight = 1
    weights = weights / weights[nonzero].mean()

    return torch.tensor(weights, dtype=torch.float32)
