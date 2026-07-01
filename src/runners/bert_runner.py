# src/runners/bert_runner.py
"""
Runner for the BERT text-classification baseline.

Pipeline:
  1. Data module setup (tokenizes cached event text, builds train/val/test splits)
  2. Model construction (pretrained transformer + classification head)
  3. Training via shared train_model()
  4. Evaluation via evaluate_model()
  5. Results persisted to RESULTS_DIR

Usage (from main.py or standalone):
    python -m src.runners.bert_runner --config path/to/bert_config.yaml
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any

import torch.nn as nn

from src.models.bert.datamodule import BertEventDataModule
from src.models.bert.model import BertForQuadClass
from src.training.train import train_model
from src.testing.evaluate import evaluate_model
from src.utils.class_weights import compute_class_weights
from src.utils.constants import NUM_QUAD_CLASSES
from src.config.paths import RESULTS_DIR, ARTIFACTS_DATA
from src.utils.experiments_logging import get_logger
from src.utils.idempotency import should_skip
from src.utils.runner_utils import collect_gpu_info, count_trainable_parameters, load_yaml_config, make_json_serializable, save_runner_results

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Core runner
# ------------------------------------------------------------------
def run_bert(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single BERT training + evaluation experiment with full result tracking.

    Parameters
    ----------
    cfg : dict
        Configuration dict (from a BERT YAML config) with at least
        "dataset", "training", "model", and "text" sections; may also carry
        "exp_id", "seed", and "split_tag" injected by main.py.

    Returns
    -------
    dict
        Either ``{"skipped": True, ...}`` if an idempotent result already
        exists for ``exp_id``, or the full results dict (metrics, confusion
        matrix, architecture, training config, and artifact paths).
    """

    start_time = time.perf_counter()
    gpu_info = collect_gpu_info(cfg["training"]["device"])

    dataset = cfg["dataset"]
    device = cfg["training"]["device"]
    exp_id = cfg.get("exp_id", "")

    logger.info(f"[BERT] Dataset={dataset}")

    # ==============================================================
    # DATA
    # ==============================================================
    dm = BertEventDataModule(
        dataset_name=dataset,
        batch_size=cfg["training"]["batch_size"],
        split_tag=cfg.get("split_tag", "default"),
        num_workers=cfg["training"].get("num_workers", 0),
        model_name=cfg["model"]["model_name"],
        max_length=cfg["text"]["max_length"],
    )
    dm.setup()

    # ==============================================================
    # MODEL
    # ==============================================================
    model = BertForQuadClass(
        num_classes=NUM_QUAD_CLASSES,
        model_name=cfg["model"]["model_name"],
        freeze_until_layer=cfg["model"]["freeze_until_layer"],
    )

    num_params = count_trainable_parameters(model)

    # ==============================================================
    # TRAINING
    # ==============================================================
    class_weights = compute_class_weights(
        dm.train_dataset.labels.cpu().numpy(), num_classes=NUM_QUAD_CLASSES
    ).to(device)

    # --- Idempotency: central check
    if exp_id:
        skip, info = should_skip(exp_id, dataset)
        if skip:
            logger.info("Skipping BERT for exp_id=%s - info=%s", exp_id, info)
            return {"skipped": True, "exp_id": exp_id, **info}

    best_model_path = train_model(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        num_epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        class_weights=class_weights,
        dataset_name=dataset,
        patience=cfg["training"]["patience"],
        lr_patience=cfg["training"].get("lr_patience", None),
        device=device,
        metric=cfg["training"].get("monitor_metric", "f1_macro"),
        exp_id=exp_id,
    )

    # ==============================================================
    # EVALUATION
    # ==============================================================
    metrics, confusion = evaluate_model(
        model=model,
        test_loader=dm.test_dataloader(),
        checkpoint_path=best_model_path,
        dataset_name=dataset,
        device=device,
        exp_id=exp_id,
    )

    runtime_sec = time.perf_counter() - start_time

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================
    results_dir = (
        RESULTS_DIR
        / dataset
        / model.__class__.__name__
    )
    results = {
        "exp_id": exp_id,
        "dataset": dataset,
        "seed": cfg.get("seed"),
        "split_tag": cfg.get("split_tag", "default"),
        "model_family": "bert",
        "model_name": cfg["model"]["model_name"],
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
            "max_length": cfg["text"]["max_length"],
            "freeze_until_layer": cfg["model"]["freeze_until_layer"],
        },
        "artifacts": {
            "best_model_path": str(best_model_path),
        },
        "metrics": make_json_serializable(metrics),
        "confusion_matrix": make_json_serializable(confusion),
        "hardware": gpu_info,
    }

    save_runner_results(results, results_dir, "bert")

    logger.info("BERT experiment completed successfully.")

    return results


# ------------------------------------------------------------------
# CLI wrapper (thin, unchanged behavior)
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse the --config CLI argument and return the parsed namespace."""
    parser = argparse.ArgumentParser(description="Run BERT on GDELT")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    """Entry point: parse CLI args, load YAML config, and dispatch to run_bert."""
    args = parse_args()
    cfg = load_yaml_config(args.config)
    run_bert(cfg)


if __name__ == "__main__":
    main()
