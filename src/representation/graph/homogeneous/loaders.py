# src/models/gnn/loaders/homogeneous.py
from typing import List, Tuple, Optional, Dict
import json

import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.representation.graph.homogeneous.builder import (
    HomogeneousEventGraphBuilder,
)
from src.models.tabular_encoder import TabularInputEncoder
from src.config.paths import ENTITIES_DATA, FEATURES_DATA, ARTIFACTS_DATA, GRAPHS_DATA
from src.config.schema.columns_schema import COLUMNS_SCHEMA
from src.utils.categorical_cardinalities import load_categorical_cardinalities
from src.utils.graph_io import save_graph


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _infer_feature_groups(df: pd.DataFrame):
    categorical_cols = []
    numeric_cols = []

    INTERNAL_COLS = {"node_idx"}

    for col in df.columns:
        if col in INTERNAL_COLS:
            continue

        schema = COLUMNS_SCHEMA.get(col)

        if schema is None:
            numeric_cols.append(col)
            continue

        if schema["kind"] in {"id", "target"}:
            continue

        if schema["kind"] == "categorical":
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return categorical_cols, numeric_cols


def _build_node_features(
    dataset_name: str,
    split: str,
    policy: str,
    num_nodes: int,
    split_tag: str = "default",
):
    """
    Returns:
        x: FloatTensor [num_nodes, feat_dim]
    """

    # --------------------------------------------------
    # Featureless mode -> constant dummy features
    # --------------------------------------------------
    if policy == "none":
        return torch.ones((num_nodes, 1), dtype=torch.float)

    if policy != "all":
        raise ValueError(f"Unknown node feature policy '{policy}'")

    feat_path = (
        FEATURES_DATA
        / dataset_name
        / f"{split}_{split_tag}_features.parquet"
    )

    df = pd.read_parquet(feat_path)

    if "GlobalEventID" not in df.columns:
        raise ValueError("Features must contain GlobalEventID")

    mapping_path = (
        ARTIFACTS_DATA
        / dataset_name
        / "graph_artifacts"
        / split_tag
        / f"{split}_node_id_map.json"
    )

    with open(mapping_path) as f:
        node_map = json.load(f)

    node_map = {int(k): int(v) for k, v in node_map.items()}

    df["node_idx"] = df["GlobalEventID"].map(node_map)

    if df["node_idx"].isna().any():
        raise ValueError(
            "Some feature rows reference nodes not present in the graph"
        )

    df = df.sort_values("node_idx")

    cat_cols, num_cols = _infer_feature_groups(df)

    x_cat: Dict[str, torch.Tensor] = {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.long)
        for col in cat_cols
    }

    x_num = (
        torch.tensor(df[num_cols].to_numpy(), dtype=torch.float)
        if num_cols
        else None
    )

    artifacts_dir = (
        ARTIFACTS_DATA
        / dataset_name
        / "features"
        / split_tag
    )

    cat_cardinalities = load_categorical_cardinalities(
        categorical_cols=cat_cols,
        artifacts_dir=artifacts_dir,
    )

    encoder = TabularInputEncoder(
        categorical_cardinalities=cat_cardinalities,
        numeric_dim=x_num.size(1) if x_num is not None else 0,
    )

    encoder.eval()
    with torch.no_grad():
        x = encoder(x_cat, x_num)

    return x


# ---------------------------------------------------------------------
# Loader builders
# ---------------------------------------------------------------------
def _build_split_loader(
    dataset_name: str,
    split: str,
    edge_keys: list[str],
    batch_size: int,
    num_neighbors: Optional[List[int]],
    node_feature_policy: str,
    shuffle: bool,
    split_tag: str = "default",
    seed: int = 42,
):
    """
    Builds a single inductive-safe homogeneous GNN loader.
    """

    builder = HomogeneousEventGraphBuilder(
        data_dir=ENTITIES_DATA,
        dataset_name=dataset_name,
        split=split,
        edge_keys=edge_keys,
        split_tag=split_tag,
        seed=seed,
    )

    graph = builder.build()

    path = (
        GRAPHS_DATA
        / dataset_name
        / "homogeneous"
        / split_tag
        / f"{split}.pt"
    )

    save_graph(graph, path)

    x = _build_node_features(
        dataset_name=dataset_name,
        split=split,
        policy=node_feature_policy,
        num_nodes=graph["num_nodes"],
        split_tag=split_tag,
    )

    data = Data(
        x=x,
        edge_index=graph["edge_index"],
        y=graph["y"],
    )

    data.num_nodes = graph["num_nodes"]

    if num_neighbors is None:
        # PyG's DataLoader collates Data into a Batch with batch_size=1
        # (number of graphs), which conflicts with NeighborLoader's convention
        # where batch_size = number of seed nodes. Using a plain wrapper avoids
        # this and keeps forward_batch slicing correct for both modes.
        class _FullBatchLoader:
            def __init__(self, d):
                self.data = d
            def __iter__(self):
                yield self.data

        loader = _FullBatchLoader(data)
    else:
        loader = NeighborLoader(
            data,
            input_nodes=torch.arange(data.num_nodes),
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return loader


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def make_gnn_loaders(
    dataset_name: str,
    edge_keys: list[str],
    batch_size: int = 1024,
    num_neighbors: Optional[List[int]] = None,
    node_feature_policy: str = "all",
    split_tag: str = "default",
    seed: int = 42,
) -> Tuple:
    train_loader = _build_split_loader(
        dataset_name,
        "train",
        edge_keys,
        batch_size,
        num_neighbors,
        node_feature_policy,
        shuffle=True,
        split_tag=split_tag,
        seed=seed,
    )

    val_entities_path = ENTITIES_DATA / dataset_name / f"valid_{split_tag}_entities.parquet"
    val_loader = (
        _build_split_loader(
            dataset_name,
            "valid",
            edge_keys,
            batch_size,
            num_neighbors,
            node_feature_policy,
            shuffle=False,
            split_tag=split_tag,
            seed=seed,
        )
        if val_entities_path.exists()
        else None
    )

    test_loader = _build_split_loader(
        dataset_name,
        "test",
        edge_keys,
        batch_size,
        num_neighbors,
        node_feature_policy,
        shuffle=False,
        split_tag=split_tag,
        seed=seed,
    )

    return train_loader, val_loader, test_loader
