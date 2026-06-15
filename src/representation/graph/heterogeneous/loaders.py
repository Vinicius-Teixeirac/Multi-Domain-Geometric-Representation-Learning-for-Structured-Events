# src/models/gnn/loaders/heterogeneous.py
from typing import List, Optional, Tuple, Union, Dict

import torch
import pandas as pd
from torch_geometric.loader import NeighborLoader

from src.representation.graph.heterogeneous.builder import (
    HeterogeneousEventGraphBuilder,
)
from src.representation.graph.heterogeneous.node_features import (
    resolve_hetero_node_features,
    build_event_node_tensors,
)
from src.config.paths import ENTITIES_DATA, FEATURES_DATA, GRAPHS_DATA
from src.utils.graph_io import save_graph


# ---------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------
def _inject_event_node_features(
    *,
    data,
    dataset_name: str,
    split: str,
    policy: str,
    split_tag: str = "default",
):
    """
    Inject tabular features + metadata into the 'event' node type
    of a HeteroData object.
    """

    if policy == "none":
        return

    if policy != "all":
        raise ValueError(f"Unknown node feature policy '{policy}'")

    # --------------------------------------------------
    # Load encoded tabular features (event-aligned)
    # --------------------------------------------------
    feat_path = (
        FEATURES_DATA
        / dataset_name
        / f"{split}_{split_tag}_features.parquet"
    )
    features_df = pd.read_parquet(feat_path)

    # --------------------------------------------------
    # Resolve feature columns
    # --------------------------------------------------
    spec = resolve_hetero_node_features(
        policy=policy,
        features_df=features_df,
    )

    if "event" not in spec:
        return

    cols = spec["event"]
    categorical_cols = cols["categorical"]
    numeric_cols = cols["numeric"]

    # --------------------------------------------------
    # Build tensors + metadata
    # --------------------------------------------------
    x_cat, cat_cardinalities, x_num = build_event_node_tensors(
        features_df=features_df,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        dataset_name=dataset_name,
        split_tag=split_tag,
    )

    # --------------------------------------------------
    # Attach to event NodeStorage (EXPLICIT)
    # --------------------------------------------------
    if x_cat:
        data["event"].x_cat = x_cat
        data["event"].x_cat_cardinalities = cat_cardinalities
        data["event"].x_cat_names = list(x_cat.keys())

    if x_num is not None:
        data["event"].x_num = x_num


# ---------------------------------------------------------------------
# Loader builder
# ---------------------------------------------------------------------
def _build_split_loader(
    dataset_name: str,
    split: str,
    batch_size: int,
    num_neighbors: Optional[Union[List[int], Dict]],
    node_feature_policy: str,
    shuffle: bool,
    split_tag: str = "default",
):
    """
    Builds a single inductive-safe heterogeneous GNN loader for one split.

    Notes:
        - Training nodes are event nodes only
        - num_neighbors may be:
            * None           -> full-batch
            * List[int]      -> same fanout for all relations
            * Dict[str, ...] -> per-relation fanouts (PyG native)
    """

    # --------------------------------------------------
    # Graph structure
    # --------------------------------------------------
    builder = HeterogeneousEventGraphBuilder(
        data_dir=ENTITIES_DATA,
        dataset_name=dataset_name,
        split=split,
        split_tag=split_tag,
    )

    data = builder.build()

    graph = builder.build()

    path = (
        GRAPHS_DATA
        / dataset_name
        / "heterogeneous"
        / split_tag
        / f"{split}.pt"
    )

    save_graph(graph, path)

    # --------------------------------------------------
    # Node features (event-only)
    # --------------------------------------------------
    _inject_event_node_features(
        data=data,
        dataset_name=dataset_name,
        split=split,
        policy=node_feature_policy,
        split_tag=split_tag,
    )

    # --------------------------------------------------
    # Loader strategy
    # --------------------------------------------------
    if num_neighbors is None:
        class _FullBatchLoader:
            def __init__(self, d):
                self.data = d
            def __iter__(self):
                yield self.data

        loader = _FullBatchLoader(data)
    else:
        input_nodes = ("event", torch.arange(data["event"].num_nodes))

        loader = NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return loader


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def make_hetero_gnn_loaders(
    dataset_name: str,
    batch_size: int = 1024,
    num_neighbors: Optional[Union[List[int], Dict]] = None,
    node_feature_policy: str = "none",
    split_tag: str = "default",
) -> Tuple:
    """
    Creates inductive heterogeneous GNN loaders.

    Node feature policies:
        - "none" -> structure-only hetero GNN
        - "all"  -> event nodes receive tabular features
    """

    train_loader = _build_split_loader(
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        node_feature_policy=node_feature_policy,
        shuffle=True,
        split_tag=split_tag,
    )

    val_entities_path = ENTITIES_DATA / dataset_name / f"valid_{split_tag}_entities.parquet"
    val_loader = (
        _build_split_loader(
            dataset_name=dataset_name,
            split="valid",
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            node_feature_policy=node_feature_policy,
            shuffle=False,
            split_tag=split_tag,
        )
        if val_entities_path.exists()
        else None
    )

    test_loader = _build_split_loader(
        dataset_name=dataset_name,
        split="test",
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        node_feature_policy=node_feature_policy,
        shuffle=False,
        split_tag=split_tag,
    )

    return train_loader, val_loader, test_loader
