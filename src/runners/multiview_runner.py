# src/runners/multiview_runner.py
"""
Runner for the MultiviewGeometricModel.

Pipeline:
  1. Data module setup (loads splits, builds actor graph, creates datasets)
  2. Model construction (uses actor graph cardinalities)
  3. Actor graph loading into model (registered buffers)
  4. model.to(device)  ← moves all parameters AND graph buffers to device
  5. Training via shared train_model()
  6. Evaluation via evaluate_model()
  7. Results persisted to RESULTS_DIR

Usage (from main.py or standalone):
    cfg = load_yaml("src/config/model_setup/multiview/multiview.yaml")
    run_multiview(cfg, dataset_name="sample_500000", split_tag="default", seed=42)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.config.paths import ARTIFACTS_DATA, RESULTS_DIR
from src.models.multiview.datamodule import MultiviewDataModule
from src.models.multiview.model import MultiviewGeometricModel
from src.testing.evaluate import evaluate_model
from src.training.train import train_model
from src.utils.class_weights import compute_class_weights
from src.utils.experiments_logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run_multiview(
    cfg: dict,
    dataset_name: str,
    split_tag: str = "default",
    seed: int = 42,
) -> dict:
    """
    Full training and evaluation pipeline for the multiview geometric model.

    Parameters
    ----------
    cfg : configuration dict (from multiview.yaml)
    dataset_name : e.g. 'sample_500000'
    split_tag : split identifier (e.g. 'default')
    seed : random seed for reproducibility

    Returns
    -------
    metrics : dict with test-set classification metrics
    """
    set_seed(seed)

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    device    = train_cfg.get("device", "cpu")

    logger.info("=== MultiviewGeometricModel | %s | seed=%d ===", dataset_name, seed)

    # ------------------------------------------------------------------
    # 1. Data module
    # ------------------------------------------------------------------
    logger.info("Setting up data module …")
    t0 = time.time()
    dm = MultiviewDataModule(
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
        "Splits — train: %d  val: %d  test: %d",
        len(dm.train_dataset),
        len(dm.valid_dataset) if dm.valid_dataset else 0,
        len(dm.test_dataset),
    )

    # ------------------------------------------------------------------
    # 2. Class weights (from training labels)
    # ------------------------------------------------------------------
    train_labels = dm.train_dataset.labels.numpy()
    class_weights_tensor = compute_class_weights(train_labels)

    # ------------------------------------------------------------------
    # 3. Model construction
    # ------------------------------------------------------------------
    logger.info("Building model …")
    model = MultiviewGeometricModel(
        actor_cardinalities=dm.actor_cardinalities,
        num_classes=dm.num_classes,
        actor_feat_embed_dim=model_cfg.get("actor_feat_embed_dim", 16),
        actor_hidden_dim=model_cfg.get("actor_hidden_dim", 128),
        actor_out_dim=model_cfg.get("actor_out_dim", 64),
        geo_hidden_dim=model_cfg.get("geo_hidden_dim", 64),
        geo_out_dim=model_cfg.get("geo_out_dim", 32),
        time_hidden_dim=model_cfg.get("time_hidden_dim", 32),
        time_out_dim=model_cfg.get("time_out_dim", 16),
        fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 128),
        num_gnn_layers=model_cfg.get("num_gnn_layers", 2),
        conv_type=model_cfg.get("conv_type", "sage"),
        dropout=model_cfg.get("dropout", 0.2),
    )

    # Load actor graph into model as registered buffers.
    # train_model() will call model.to(device) and move the buffers too.
    model.set_actor_graph(dm.actor_graph.x, dm.actor_graph.edge_index)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # ------------------------------------------------------------------
    # 4. Training  (model.to(device) is called inside train_model)
    # ------------------------------------------------------------------
    logger.info("Training …")
    train_model(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        num_epochs=train_cfg.get("epochs", 200),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        patience=train_cfg.get("patience", 30),
        class_weights=class_weights_tensor,
        dataset_name=dataset_name,
        device=device,
        metric=train_cfg.get("metric", "f1_macro"),
        exp_id=f"multiview_s{seed}",
    )

    # ------------------------------------------------------------------
    # 5. Evaluation  (loads the best checkpoint saved by train_model)
    # ------------------------------------------------------------------
    exp_id = f"multiview_s{seed}"
    checkpoint_path = (
        ARTIFACTS_DATA
        / dataset_name
        / "models"
        / "MultiviewGeometricModel"
        / exp_id
        / "best_model.pt"
    )

    logger.info("Evaluating on test set …")
    metrics, _ = evaluate_model(
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
    results_dir = Path(RESULTS_DIR) / dataset_name / "multiview" / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "model": "multiview_geometric",
                "dataset": dataset_name,
                "split_tag": split_tag,
                "seed": seed,
                "conv_type": model_cfg.get("conv_type", "sage"),
                "num_gnn_layers": model_cfg.get("num_gnn_layers", 2),
                "actor_nodes": dm.actor_graph.num_nodes,
                "actor_edges": int(dm.actor_graph.edge_index.shape[1]),
                "n_params": n_params,
                **metrics,
            },
            f,
            indent=2,
        )
    logger.info("Results saved -> %s", results_path)
    return metrics
