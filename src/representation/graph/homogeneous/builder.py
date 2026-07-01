"""Constructs the homogeneous (single node-type) event graph.

Builds one graph per split from the entity parquets: each event is a node,
and events sharing an entity key (actor, geo, day) are linked via
build_binary_edges_from_shared_keys. No node features are attached here;
they are loaded separately by homogeneous/loaders.py.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data


from src.representation.graph.base import GraphBuilder
from src.utils.loading import load_parquet
from src.representation.graph.indexing import add_node_index
from src.representation.graph.homogeneous.edge_rules import (
    build_binary_edges_from_shared_keys,
)
from src.config.paths import ARTIFACTS_DATA


class HomogeneousEventGraphBuilder(GraphBuilder):
    """
    Builds a homogeneous EVENT graph (event-inductive).

    Nodes:
        - one node per event instance

    Edges:
        - undirected
        - events connected if they share at least one entity key

    Notes:
        - No node features attached here
        - Tabular features handled separately
    """

    def __init__(
        self,
        data_dir: Path,
        dataset_name: str,
        split: str,
        edge_keys: List[str],
        *,
        split_tag: str = "default",
        node_id_col: str = "GlobalEventID",
        label_col: str = "QuadClass",
        max_neighbors_per_key: Optional[Dict[str, int]] = None,
        default_max_neighbors: int = 10,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        data_dir : Path
            Root directory containing entity parquets (ENTITIES_DATA).
        dataset_name : str
            Dataset subdirectory name.
        split : str
            Split name ('train', 'valid', or 'test').
        edge_keys : list[str]
            Entity columns used to build shared-key edges
            (e.g. ['Actor1ID', 'Actor2ID', 'Event_GeoID']).
        split_tag : str
            Split regime identifier.
        node_id_col : str
            Column whose values become node identities.
        label_col : str
            Column with integer class labels.
        max_neighbors_per_key : dict or None
            Per-key neighbour sampling limit. Falls back to
            default_max_neighbors when a key is absent.
        default_max_neighbors : int
            Default per-node neighbour cap when no key-specific limit is set.
        seed : int
            Random seed for reproducible neighbour sampling.
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.split_tag = split_tag

        self.edge_keys = edge_keys
        self.node_id_col = node_id_col
        self.label_col = label_col

        self.max_neighbors_per_key = max_neighbors_per_key
        self.default_max_neighbors = default_max_neighbors
        self.seed = seed

    def build(self) -> Dict[str, Any]:
        """
        Load entity parquet, construct the homogeneous event graph, and return it.

        Returns
        -------
        torch_geometric.data.Data
            Graph with edge_index, y (labels), and num_nodes set.
            No node feature matrix is attached; features are loaded separately.
        """
        # --------------------------------------------------
        # Load split
        # --------------------------------------------------
        
        entities_dir = self.data_dir / self.dataset_name
        df: pd.DataFrame = load_parquet(f"{self.split}_{self.split_tag}_entities.parquet", entities_dir)

        # --------------------------------------------------
        # Node indexing (deterministic, per split)
        # --------------------------------------------------
        mapping = add_node_index(
            df,
            id_col=self.node_id_col,
            index_col="node_idx",
        )

        artifacts_dir = ARTIFACTS_DATA / self.dataset_name / "graph_artifacts" / self.split_tag
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        mapping_path = artifacts_dir / f"{self.split}_node_id_map.json"

        mapping_json = {str(k): int(v) for k, v in mapping.items()}

        with open(mapping_path, "w") as f:
            json.dump(mapping_json, f, indent=2)

        num_nodes = df.shape[0]

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------
        y = torch.tensor(
            df[self.label_col].to_numpy(dtype="int64"),
            dtype=torch.long,
        )

        # --------------------------------------------------
        # Graph structure
        # --------------------------------------------------
        src, dst = build_binary_edges_from_shared_keys(
            df=df,
            keys=self.edge_keys,
            node_idx_col="node_idx",
            max_neighbors_per_key=self.max_neighbors_per_key,
            default_max_neighbors=self.default_max_neighbors,
            seed=self.seed,
        )


        edge_index = torch.from_numpy(np.vstack((src, dst))).long()

        # --------------------------------------------------
        # Output
        # --------------------------------------------------
        data = Data(
            edge_index=edge_index,
            y=y,
            num_nodes=num_nodes,
        )
        return data
