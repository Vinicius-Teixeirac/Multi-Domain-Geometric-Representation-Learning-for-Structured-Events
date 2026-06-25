# src/utils/seed.py
import random

import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Fix all PRNG seeds for reproducible experiment runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # covers multi-GPU; no-op on CPU-only builds