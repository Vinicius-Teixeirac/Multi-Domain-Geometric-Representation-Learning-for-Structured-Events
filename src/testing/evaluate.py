# src/testing/evaluate.py
"""
Central evaluation module.

Provides ``evaluate_model``, the shared test-set inference routine
invoked by every model family runner (MLP, GNN, BERT, multi-domain
geometric model). Loads the best checkpoint saved by ``train_model``,
runs inference, computes classification metrics and a confusion matrix,
and persists both to ARTIFACTS_DATA. Also runs a one-time compute/latency
profile (see src/utils/profiling.py) so every run's results carry exact
per-sample GFLOPs and throughput/latency alongside its metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import (
    compute_classification_metrics,
    compute_confusion,
)
from src.utils.profiling import profile_model
from src.config.paths import ARTIFACTS_DATA


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: Path,
    dataset_name: str,
    device: str = "cpu",
    exp_id: str = "",
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
    """Load the best checkpoint, run inference on the test set, and persist metrics.

    Returns (metrics dict, confusion matrix ndarray, compute profile dict).
    """
    # Load to CPU and discard optimizer state; eval only needs model weights.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.to(device)
    model.eval()

    all_logits, all_targets = [], []
    profile_batch = None

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if profile_batch is None:
                profile_batch = batch
            logits, targets = cast(Any, model).forward_batch(batch, device)
            all_logits.append(logits)
            all_targets.append(targets)

    logits = torch.cat(all_logits).cpu()
    targets = torch.cat(all_targets).cpu()

    metrics = compute_classification_metrics(logits, targets)
    conf = compute_confusion(logits, targets)
    assert profile_batch is not None, "test_loader yielded no batches"
    profile = profile_model(model, profile_batch, device)

    out_dir = ARTIFACTS_DATA / dataset_name / "models" / model.__class__.__name__ / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(conf, out_dir / "confusion_matrix.pt")

    return metrics, conf, profile
