# src/runners/gnn_runner.py
"""
Runner for the GNN model family (homogeneous and heterogeneous graphs).

Pipeline:
  1. Loader construction (builds the actor/event graph and mini-batch
     neighbor-sampling loaders for the configured graph type)
  2. Model construction (conv type, node-feature policy, and graph type
     jointly determine the architecture branch taken)
  3. Training via shared train_model()
  4. Evaluation via evaluate_model()
  5. Results persisted to RESULTS_DIR

Usage (from main.py or standalone):
    cfg = load_yaml_config("path/to/gnn_config.yaml")
    run_gnn(cfg)
"""

import time
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn

from src.training.train import train_model
from src.testing.evaluate import evaluate_model
from src.utils.class_weights import compute_class_weights
from src.utils.constants import NUM_QUAD_CLASSES
from src.config.paths import RESULTS_DIR, ARTIFACTS_DATA
from src.utils.idempotency import should_skip

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
from src.models.gnn.homogeneous import HomogeneousGNN
from src.models.gnn.heterogeneous import HeterogeneousGNN

# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
from src.representation.graph.homogeneous.loaders import make_gnn_loaders
from src.representation.graph.heterogeneous.loaders import make_hetero_gnn_loaders

# ------------------------------------------------------------------
# Encoders
# ------------------------------------------------------------------
from src.models.tabular_encoder import TabularInputEncoder

from src.utils.experiments_logging import get_logger
from src.utils.runner_utils import collect_gpu_info, count_trainable_parameters, make_json_serializable, save_runner_results

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def run_gnn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single GNN experiment (homogeneous or heterogeneous).

    Dispatches on ``cfg["graph"]["type"]`` ("homogeneous" or
    "heterogeneous") and ``cfg["graph"]["node_features"]`` ("all" or
    "none"), each combination selecting a different model-construction
    branch (plain GNN vs. tabular-encoder-fed GNN, homogeneous vs.
    relation-aware heterogeneous convolutions).

    Parameters
    ----------
    cfg : dict
        Configuration dict (from a GNN YAML config) with at least
        "dataset", "model", "training", and "graph" sections; may also
        carry "exp_id", "seed", and "split_tag" injected by main.py.

    Returns
    -------
    dict
        Either ``{"skipped": True, ...}`` if an idempotent result already
        exists for ``exp_id``, or the full results dict (metrics, confusion
        matrix, architecture, training config, and artifact paths).

    Raises
    ------
    ValueError
        If ``model_name == "gin"`` is combined with a featureless graph
        (GINConv requires node features), or if ``graph_type`` /
        ``node_feature_policy`` is not one of the supported values.
    """

    start_time = time.perf_counter()
    gpu_info = collect_gpu_info(cfg["training"]["device"])

    dataset = cfg["dataset"]
    model_name = cfg["model"]["name"]
    device = cfg["training"]["device"]
    graph_type = cfg["graph"]["type"]
    node_feature_policy = cfg["graph"]["node_features"]

    logger.info(
        f"[GNN] Dataset={dataset} | Graph={graph_type} | Model={model_name} | "
        f"NodeFeatures={node_feature_policy}"
    )

    exp_id = cfg.get("exp_id", "")
    split_tag = cfg.get("split_tag", "default")
    seed = cfg.get("seed", 42)  # repo-wide reproducibility default (see main.py --seed)

    # GINConv aggregates neighbor node features via a learned sum-based update
    # (Xu et al., 2019); with no node features there is nothing for it to
    # aggregate, so the featureless ("none") policy is architecturally
    # incompatible with GIN. GAT/GraphSAGE can fall back to learned embeddings
    # instead, so they remain valid without node features.
    if node_feature_policy == "none" and model_name == "gin":
        raise ValueError(
            "GINConv does not support featureless graphs. "
            "Use GAT or GraphSAGE, or enable node features."
        )

    # --- Idempotency: central check (before any expensive setup)
    if exp_id:
        skip, info = should_skip(exp_id, dataset)
        if skip:
            logger.info("Skipping GNN for exp_id=%s - info=%s", exp_id, info)
            return {"skipped": True, "exp_id": exp_id, **info}

    # ==============================================================
    # DATA
    # ==============================================================

    if graph_type == "homogeneous":
        train_loader, val_loader, test_loader = make_gnn_loaders(
            dataset_name=dataset,
            edge_keys=cfg["graph"]["edge_keys"],
            batch_size=cfg["training"]["batch_size"],
            num_neighbors=cfg["graph"].get("num_neighbors"),
            node_feature_policy=node_feature_policy,
            split_tag=split_tag,
            seed=seed,
        )

        data = train_loader.data

        # ------------------------------
        # Model construction
        # ------------------------------
        if node_feature_policy == "all":
            in_channels = data.x.size(1)

            model = HomogeneousGNN(
                conv_type=cfg["model"]["name"],
                in_channels=in_channels,
                hidden_channels=cfg["model"]["hidden_dim"],
                out_channels=NUM_QUAD_CLASSES,
                num_layers=cfg["model"]["num_layers"],
                dropout=cfg["model"]["dropout"],
                heads=cfg["model"].get("heads", 1),
            )

            architecture = {
                "input_dim": in_channels,
                "hidden_dim": cfg["model"]["hidden_dim"],
                "num_layers": cfg["model"]["num_layers"],
                "heads": cfg["model"].get("heads", 1),
                "dropout": cfg["model"]["dropout"],
            }

        elif node_feature_policy == "none":
            model = HomogeneousGNN(
                conv_type=cfg["model"]["name"],
                in_channels=0,
                hidden_channels=cfg["model"]["hidden_dim"],
                out_channels=NUM_QUAD_CLASSES,
                num_layers=cfg["model"]["num_layers"],
                dropout=cfg["model"]["dropout"],
                heads=cfg["model"].get("heads", 1),
            )

            architecture = {
                "input_dim": "no features",
                "hidden_dim": cfg["model"]["hidden_dim"],
                "num_layers": cfg["model"]["num_layers"],
                "heads": cfg["model"].get("heads", 1),
                "dropout": cfg["model"]["dropout"],
            }


        else:
            raise ValueError(
                f"Unknown node_feature_policy '{node_feature_policy}'"
            )

        class_weights = compute_class_weights(
            data.y.cpu().numpy(), num_classes=NUM_QUAD_CLASSES
        ).to(device)

    # ==============================================================
    # HETEROGENEOUS
    # ==============================================================

    elif graph_type == "heterogeneous":
        train_loader, val_loader, test_loader = make_hetero_gnn_loaders(
            dataset_name=dataset,
            batch_size=cfg["training"]["batch_size"],
            num_neighbors=cfg["graph"].get("num_neighbors"),
            node_feature_policy=node_feature_policy,
            split_tag=split_tag,
        )

        data = train_loader.data
        metadata = data.metadata()

        if node_feature_policy == "all":
            encoder = TabularInputEncoder(
                categorical_cardinalities=data["event"].x_cat_cardinalities,
                numeric_dim=(
                    data["event"].x_num.size(1)
                    if data["event"].x_num is not None
                    else 0
                ),
                embedding_dim_rule=cfg["model"].get(
                    "embedding_dim_rule", "sqrt"
                ),
                embedding_dropout=cfg["model"].get(
                    "embedding_dropout", 0.2
                ),
            )

            model = HeterogeneousGNN(
                conv_type=cfg["model"]["name"],
                in_channels=encoder.output_dim,
                hidden_channels=cfg["model"]["hidden_dim"],
                out_channels=NUM_QUAD_CLASSES,
                metadata=metadata,
                num_relations=len(metadata[1]),
                num_layers=cfg["model"]["num_layers"],
                heads=cfg["model"].get("heads", 1),
                dropout=cfg["model"]["dropout"],
                encoder=encoder,
                event_type="event",
            )

            architecture = {
                "encoder_output_dim": encoder.output_dim,
                "hidden_dim": cfg["model"]["hidden_dim"],
                "num_layers": cfg["model"]["num_layers"],
                "heads": cfg["model"].get("heads", 1),
                "dropout": cfg["model"]["dropout"],
                "num_relations": len(metadata[1]),
            }

        elif node_feature_policy == "none":
            num_nodes_per_type = {
                ntype: data[ntype].num_nodes
                for ntype in data.node_types
            }

            model = HeterogeneousGNN(
                conv_type=cfg["model"]["name"],
                in_channels=0,
                hidden_channels=cfg["model"]["hidden_dim"],
                out_channels=NUM_QUAD_CLASSES,
                metadata=metadata,
                num_relations=len(metadata[1]),
                num_layers=cfg["model"]["num_layers"],
                heads=cfg["model"].get("heads", 1),
                dropout=cfg["model"]["dropout"],
                encoder=None,
                event_type="event",
                num_nodes_per_type=num_nodes_per_type,
            )

            architecture = {
                "encoder": None,
                "hidden_dim": cfg["model"]["hidden_dim"],
                "num_layers": cfg["model"]["num_layers"],
                "heads": cfg["model"].get("heads", 1),
                "dropout": cfg["model"]["dropout"],
                "num_relations": len(metadata[1]),
            }

        else:
            raise ValueError(
                f"Unknown node_feature_policy '{node_feature_policy}'"
            )

        class_weights = compute_class_weights(
            data["event"].y.cpu().numpy(), num_classes=NUM_QUAD_CLASSES
        ).to(device)

    else:
        raise ValueError(f"Unknown graph type '{graph_type}'")

    num_params = count_trainable_parameters(model)

    # ==============================================================
    # TRAINING
    # ==============================================================

    best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
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
        test_loader=test_loader,
        checkpoint_path=best_model_path,
        dataset_name=dataset,
        device=device,
        exp_id=exp_id,
    )

    runtime_sec = time.perf_counter() - start_time

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================

    results_dir = RESULTS_DIR / dataset / model.__class__.__name__
    results = {
        "exp_id": exp_id,
        "dataset": dataset,
        "seed": seed,
        "split_tag": split_tag,
        "model_family": "gnn",
        "model_name": cfg["model"]["name"],
        "graph_type": graph_type,
        "node_feature_policy": node_feature_policy,
        "num_parameters": num_params,
        "runtime_seconds": round(runtime_sec, 3),
        "training": {
            "batch_size": cfg["training"]["batch_size"],
            "epochs": cfg["training"]["epochs"],
            "patience": cfg["training"]["patience"],
            "lr": cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "monitor_metric": cfg["training"].get(
                "monitor_metric", "f1_macro"
            ),
        },
        "architecture": architecture,
        "artifacts": {
            "best_model_path": str(best_model_path),
        },
        "metrics": make_json_serializable(metrics),
        "confusion_matrix": make_json_serializable(confusion),
        "hardware": gpu_info,
    }

    save_runner_results(results, results_dir, "gnn")

    logger.info("GNN experiment completed successfully.")

    return results
