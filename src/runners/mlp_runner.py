# src/runners/mlp_runner.py

import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np

from src.models.mlp.datamodule import EventDataModule
from src.models.mlp.model import EventMLP
from src.training.train import train_model
from src.testing.evaluate import evaluate_model
from src.utils.class_weights import compute_class_weights
from src.config.paths import ARTIFACTS_DATA, RESULTS_DIR
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------
def run_mlp(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single MLP training + evaluation pipeline.
    """

    start_time = time.perf_counter()

    # ----------------------
    # Data
    # ----------------------
    dm = EventDataModule(
        dataset_name=cfg["dataset"],
        split_tag=cfg.get("split_tag", "default"),
        batch_size=cfg["training"]["batch_size"],
    )
    dm.setup()

    # ----------------------
    # Model
    # ----------------------
    model = EventMLP(
        categorical_cardinalities=dm.categorical_cardinalities,
        numeric_dim=len(dm.numeric_cols),
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=dm.num_classes,
    )

    num_params = count_trainable_parameters(model)

    # ----------------------
    # Training
    # ----------------------
    class_weights = compute_class_weights(
        dm.train_df["QuadClass"].to_numpy()
    ).to(cfg["training"]["device"])

    exp_id = cfg.get("exp_id", "")

    # --- Idempotency: skip if checkpoint or results already exist for this exp_id
    if exp_id:
        model_name = model.__class__.__name__
        best_ckpt = ARTIFACTS_DATA / cfg["dataset"] / "models" / model_name / exp_id / "best_model.pt"
        if best_ckpt.exists():
            logger.info("Checkpoint exists for exp_id=%s at %s — skipping training", exp_id, best_ckpt)
            # Try to find existing results JSON (optional)
            results_dir = RESULTS_DIR / cfg["dataset"] / model_name
            existing = None
            if results_dir.exists():
                for p in results_dir.glob("*.json"):
                    try:
                        txt = p.read_text()
                        if exp_id in txt:
                            existing = p
                            break
                    except Exception:
                        continue
            return {"skipped": True, "exp_id": exp_id, "checkpoint": str(best_ckpt), "results_file": str(existing) if existing is not None else None}

    best_model_path = train_model(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        num_epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        class_weights=class_weights,
        dataset_name=cfg["dataset"],
        patience=cfg["training"]["patience"],
        device=cfg["training"]["device"],
        metric=cfg["training"].get("monitor_metric", "f1_macro"),
        exp_id=exp_id,
    )

    # ----------------------
    # Evaluation
    # ----------------------
    metrics, confusion = evaluate_model(
        model=model,
        test_loader=dm.test_dataloader(),
        checkpoint_path=best_model_path,
        dataset_name=cfg["dataset"],
        device=cfg["training"]["device"],
        exp_id=exp_id,
    )

    runtime_sec = time.perf_counter() - start_time

    # ----------------------
    # Save results
    # ----------------------
    results_dir = (
        RESULTS_DIR
        / cfg["dataset"]
        / model.__class__.__name__
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "exp_id": exp_id,
        "dataset": cfg["dataset"],
        "seed": cfg.get("seed"),
        "split_tag": cfg.get("split_tag", "default"),
        "model_family": "mlp",
        "model_name": "EventMLP",
        "num_parameters": num_params,
        "runtime_seconds": round(runtime_sec, 3),
        "training": {
            "batch_size": cfg["training"]["batch_size"],
            "epochs": cfg["training"]["epochs"],
            "patience": cfg["training"]["patience"],
            "lr": cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "monitor_metric": cfg["training"].get("monitor_metric", "f1_macro"),
        },
        "architecture": {
            "hidden_dims": cfg["model"]["hidden_dims"],
            "categorical_cardinalities": dm.categorical_cardinalities,
            "numeric_dim": len(dm.numeric_cols),
        },
        "artifacts": {
            "best_model_path": str(best_model_path),
        },
        "metrics": make_json_serializable(metrics),
        "confusion_matrix": make_json_serializable(confusion),
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = results_dir / f"mlp_results_{timestamp}.json"

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP on GDELT")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_mlp(cfg)
    print("\nMLP run completed successfully.")


if __name__ == "__main__":
    main()
