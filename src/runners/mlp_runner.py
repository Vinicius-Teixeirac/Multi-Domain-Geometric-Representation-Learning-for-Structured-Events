# src/runners/mlp_runner.py

import argparse
import torch.nn as nn
import time
from pathlib import Path
from typing import Dict, Any

from src.models.mlp.datamodule import EventDataModule
from src.models.mlp.model import EventMLP
from src.training.train import train_model
from src.testing.evaluate import evaluate_model
from src.utils.class_weights import compute_class_weights
from src.utils.constants import NUM_QUAD_CLASSES
from src.config.paths import ARTIFACTS_DATA, RESULTS_DIR
from src.utils.experiments_logging import get_logger
from src.utils.idempotency import should_skip
from src.utils.runner_utils import collect_gpu_info, count_trainable_parameters, load_yaml_config, make_json_serializable, save_runner_results

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------
def run_mlp(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single MLP training + evaluation pipeline.
    """

    start_time = time.perf_counter()
    gpu_info = collect_gpu_info(cfg["training"]["device"])

    exp_id = cfg.get("exp_id", "")

    # --- Idempotency: central check (before any expensive setup)
    if exp_id:
        skip, info = should_skip(exp_id, cfg["dataset"])
        if skip:
            logger.info("Skipping MLP for exp_id=%s - info=%s", exp_id, info)
            return {"skipped": True, "exp_id": exp_id, **info}

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
        dm.train_df["QuadClass"].to_numpy(), num_classes=NUM_QUAD_CLASSES
    ).to(cfg["training"]["device"])

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
        lr_patience=cfg["training"].get("lr_patience", None),
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
        "hardware": gpu_info,
    }

    save_runner_results(results, results_dir, "mlp")

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


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    run_mlp(cfg)
    print("\nMLP run completed successfully.")


if __name__ == "__main__":
    main()
