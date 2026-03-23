# src/runners/gnn_runner.py

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
from torch.nn.parameter import UninitializedParameter

from src.training.train import train_model
from src.testing.evaluate import evaluate_model
from src.utils.class_weights import compute_class_weights
from src.config.paths import RESULTS_DIR, ARTIFACTS_DATA

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

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def count_trainable_parameters(model):
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, UninitializedParameter):
            continue
        total += p.numel()
    return total


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def run_gnn(cfg: Dict) -> Dict:
    """
    Run a single GNN experiment (homogeneous or heterogeneous).
    """

    start_time = time.perf_counter()

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
    seed = cfg.get("seed", 42)

    if node_feature_policy == "none" and model_name == "gin":
        raise ValueError(
            "GINConv does not support featureless graphs. "
            "Use GAT or GraphSAGE, or enable node features."
        )


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
                out_channels=cfg["model"]["num_classes"],
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
                out_channels=cfg["model"]["num_classes"],
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
            data.y.cpu().numpy()
        ).to(device)

    # ==============================================================
    # HETEROGENEOUS
    # ==============================================================

    elif graph_type == "heterogeneous":
        train_loader, val_loader, test_loader = make_hetero_gnn_loaders(
            dataset_name=dataset,
            batch_size=cfg["training"]["batch_size"],
            num_neighbors=cfg["graph"]["num_neighbors"],
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
                out_channels=cfg["model"]["num_classes"],
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
                out_channels=cfg["model"]["num_classes"],
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
            data["event"].y.cpu().numpy()
        ).to(device)

    else:
        raise ValueError(f"Unknown graph type '{graph_type}'")

    num_params = count_trainable_parameters(model)

    # ==============================================================
    # TRAINING
    # ==============================================================

    # --- Idempotency: skip if checkpoint or results already exist for this exp_id
    if exp_id:
        model_name = None
        # infer model class name
        if graph_type == "homogeneous":
            model_name = model.__class__.__name__
        else:
            model_name = model.__class__.__name__

        best_ckpt = ARTIFACTS_DATA / dataset / "models" / model_name / exp_id / "best_model.pt"
        if best_ckpt.exists():
            logger.info("Checkpoint exists for exp_id=%s at %s — skipping training", exp_id, best_ckpt)
            results_dir = RESULTS_DIR / dataset / model_name
            existing = None
            if results_dir.exists():
                for p in results_dir.glob("*.json"):
                    try:
                        if exp_id in p.read_text():
                            existing = p
                            break
                    except Exception:
                        continue
            return {"skipped": True, "exp_id": exp_id, "checkpoint": str(best_ckpt), "results_file": str(existing) if existing is not None else None}

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
    results_dir.mkdir(parents=True, exist_ok=True)

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
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = results_dir / f"gnn_results_{timestamp}.json"

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("GNN experiment completed successfully.")

    return results
