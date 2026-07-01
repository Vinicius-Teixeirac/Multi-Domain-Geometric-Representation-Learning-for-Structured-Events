# src/runners/multi_domain_runner.py
"""
Runner for the MultiDomainGeometricModel.

Pipeline:
  1. Data module setup (loads splits, builds actor graph, creates datasets)
  2. Model construction (uses actor graph cardinalities)
  3. Actor graph loading into model (registered buffers)
  4. model.to(device)  <- moves all parameters AND graph buffers to device
  5. Training via shared train_model()
  6. Evaluation via evaluate_model()
  7. Results persisted to RESULTS_DIR

Usage (from main.py or standalone):
    cfg = load_yaml("src/config/model_setup/multi_domain/multi_domain.yaml")
    run_multi_domain(cfg, dataset_name="sample_500000", split_tag="default", seed=42)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch

from src.config.paths import ARTIFACTS_DATA, RESULTS_DIR
from src.models.multi_domain.datamodule import MultiDomainDataModule
from src.models.multi_domain.model import MultiDomainGeometricModel
from src.testing.evaluate import evaluate_model
from src.training.train import train_model
from src.utils.class_weights import compute_class_weights
from src.utils.constants import NUM_QUAD_CLASSES
from src.utils.experiments_logging import get_logger
from src.utils.idempotency import should_skip
from src.utils.runner_utils import collect_gpu_info, count_trainable_parameters, make_json_serializable, save_runner_results
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run_multi_domain(
    cfg: dict,
    dataset_name: str,
    split_tag: str = "default",
    seed: int = 42,
) -> dict:
    """
    Full training and evaluation pipeline for the multi-domain geometric model.

    Parameters
    ----------
    cfg : configuration dict (from multi_domain.yaml)
    dataset_name : e.g. 'sample_500000'
    split_tag : split identifier (e.g. 'default')
    seed : random seed for reproducibility

    Returns
    -------
    metrics : dict with test-set classification metrics
    """
    set_seed(seed)
    start_time = time.perf_counter()
    gpu_info = collect_gpu_info(cfg["training"].get("device", "cpu"))

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    device    = train_cfg.get("device", "cpu")

    logger.info("=== MultiDomainGeometricModel | %s | seed=%d ===", dataset_name, seed)

    config_stem = Path(cfg.get("_config_path", "unknown")).stem
    descriptor = f"{dataset_name}_{split_tag}_{config_stem}_seed{seed}"
    # 8 hex chars (32 bits) is enough collision resistance to distinguish
    # experiment configs in this run's namespace - this is not a security
    # or cryptographic use of MD5, just a short, stable experiment ID.
    exp_id = "multi_domain_" + hashlib.md5(descriptor.encode()).hexdigest()[:8]

    # --- Idempotency: central check (before any expensive setup)
    skip, info = should_skip(exp_id, dataset_name)
    if skip:
        logger.info("Skipping multi-domain run for exp_id=%s - info=%s", exp_id, info)
        return {"skipped": True, "exp_id": exp_id, **info}

    # ------------------------------------------------------------------
    # 1. Data module
    # ------------------------------------------------------------------
    logger.info("Setting up data module ...")
    t0 = time.time()
    dm = MultiDomainDataModule(
        dataset_name=dataset_name,
        split_tag=split_tag,
        batch_size=train_cfg.get("batch_size", 512),
        num_workers=train_cfg.get("num_workers", 0),
    )
    dm.setup()
    logger.info(
        "Actor graph: %d nodes, %d edges  (built in %.1fs)",
        dm.actor_graph.num_nodes,
        dm.actor_graph.edge_index.shape[1],
        time.time() - t0,
    )
    logger.info(
        "Splits - train: %d  val: %d  test: %d",
        len(dm.train_dataset),
        len(dm.valid_dataset) if dm.valid_dataset else 0,
        len(dm.test_dataset),
    )

    # ------------------------------------------------------------------
    # 2. Class weights (from training labels)
    # ------------------------------------------------------------------
    train_labels = dm.train_dataset.labels.numpy()
    class_weights_tensor = compute_class_weights(train_labels, num_classes=NUM_QUAD_CLASSES).to(device)

    # ------------------------------------------------------------------
    # 3. Model construction
    # ------------------------------------------------------------------
    logger.info("Building model ...")
    model = MultiDomainGeometricModel(
        model_cfg=model_cfg,
        actor_cardinalities=dm.actor_cardinalities,
        geo_country_cardinality=dm.geo_country_cardinality,
        num_classes=dm.num_classes,
    )

    # Load actor graph into model as registered buffers.
    # train_model() will call model.to(device) and move the buffers too.
    model.set_actor_graph(
        dm.actor_graph.x,
        dm.actor_graph.edge_index,
        graph_edge_attr=getattr(dm.actor_graph, "edge_attr", None),
    )

    n_params = count_trainable_parameters(model)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # ------------------------------------------------------------------
    # 4. Training  (model.to(device) is called inside train_model)
    # ------------------------------------------------------------------
    logger.info("Training ...")
    best_model_path = train_model(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        num_epochs=train_cfg.get("epochs", 200),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        patience=train_cfg.get("patience", 30),
        lr_patience=train_cfg.get("lr_patience", None),
        class_weights=class_weights_tensor,
        dataset_name=dataset_name,
        device=device,
        metric=train_cfg.get("monitor_metric", "f1_macro"),
        exp_id=exp_id,
    )

    # ------------------------------------------------------------------
    # 5. Evaluation  (loads the best checkpoint saved by train_model)
    # ------------------------------------------------------------------
    checkpoint_path = best_model_path

    logger.info("Evaluating on test set ...")
    metrics, confusion = evaluate_model(
        model=model,
        test_loader=dm.test_dataloader(),
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        device=device,
        exp_id=exp_id,
    )

    # ------------------------------------------------------------------
    # 6. Persist summary results
    # ------------------------------------------------------------------
    runtime_sec = time.perf_counter() - start_time

    results_dir = RESULTS_DIR / dataset_name / model.__class__.__name__
    results = {
        "exp_id": exp_id,
        "dataset": dataset_name,
        "seed": seed,
        "split_tag": split_tag,
        "model_family": "multi_domain",
        "model_name": model.__class__.__name__,
        "num_parameters": n_params,
        "runtime_seconds": round(runtime_sec, 3),
        "training": {
            "batch_size": train_cfg.get("batch_size", 512),
            "epochs": train_cfg.get("epochs", 200),
            "patience": train_cfg.get("patience", 30),
            "lr": train_cfg.get("lr", 1e-3),
            "weight_decay": train_cfg.get("weight_decay", 1e-4),
            "monitor_metric": train_cfg.get("monitor_metric", "f1_macro"),
        },
        "architecture": {
            "actor":    model_cfg["actor"],
            "geo":      model_cfg["geo"],
            "temporal": model_cfg["temporal"],
            "fusion":   model_cfg["fusion"],
            "actor_nodes": dm.actor_graph.num_nodes,
            "actor_edges": int(dm.actor_graph.edge_index.shape[1]),
        },
        "artifacts": {
            "best_model_path": str(checkpoint_path),
        },
        "metrics": make_json_serializable(metrics),
        "confusion_matrix": make_json_serializable(confusion),
        "hardware": gpu_info,
    }

    results_path = save_runner_results(results, results_dir, "multi_domain")

    logger.info("Results saved -> %s", results_path)
    return results
