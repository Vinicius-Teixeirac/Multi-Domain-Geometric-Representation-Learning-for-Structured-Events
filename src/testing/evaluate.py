# src/testing/evaluate.py
import json
from pathlib import Path
import torch
from tqdm import tqdm

from src.utils.metrics import (
    compute_classification_metrics,
    compute_confusion,
)
from src.config.paths import ARTIFACTS_DATA


def evaluate_model(
    model,
    test_loader,
    checkpoint_path: Path,
    dataset_name: str,
    device: str = "cpu",
    exp_id: str = "",
):
    # weights_only=False because checkpoints include optimizer state dict (non-tensor objects)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_logits, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits, targets = model.forward_batch(batch, device)
            all_logits.append(logits)
            all_targets.append(targets)

    logits = torch.cat(all_logits).cpu()
    targets = torch.cat(all_targets).cpu()

    metrics = compute_classification_metrics(logits, targets)
    conf = compute_confusion(logits, targets)

    out_dir = ARTIFACTS_DATA / dataset_name / "models" / model.__class__.__name__ / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(conf, out_dir / "confusion_matrix.pt")

    return metrics, conf
