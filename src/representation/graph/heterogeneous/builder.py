# src/representation/graph/heterogeneous/builder.py
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.representation.graph.base import GraphBuilder
from src.utils.loading import load_parquet
from src.representation.graph.indexing import add_node_index

from .edge_rules import build_event_component_edges


class HeterogeneousEventGraphBuilder(GraphBuilder):
    """
    Builds a heterogeneous EVENT graph (event-inductive, split-wise).
    """

    EVENT = "event"
    ACTOR1 = "actor1"
    ACTOR2 = "actor2"
    GEO = "geo"
    DAY = "day"

    HAS_ACTOR1 = (EVENT, "has_actor1", ACTOR1)
    HAS_ACTOR2 = (EVENT, "has_actor2", ACTOR2)
    OCCURRED_IN_GEO = (EVENT, "occurred_in_geo", GEO)
    OCCURRED_ON_DAY = (EVENT, "occurred_on_day", DAY)

    REV_HAS_ACTOR1 = (ACTOR1, "rev_has_actor1", EVENT)
    REV_HAS_ACTOR2 = (ACTOR2, "rev_has_actor2", EVENT)
    REV_IN_GEO = (GEO, "rev_occurred_in_geo", EVENT)
    REV_ON_DAY = (DAY, "rev_occurred_on_day", EVENT)

    def __init__(
        self,
        data_dir: Path,
        dataset_name: str,
        split: str,
        *,
        split_tag: str = "default",
        node_id_col: str = "GlobalEventID",
        label_col: str = "QuadClass",
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.split_tag = split_tag

        self.node_id_col = node_id_col
        self.label_col = label_col

    def build(self) -> HeteroData:
        # --------------------------------------------------
        # Load split
        # --------------------------------------------------
        # df = load_split(
        #     data_dir=self.data_dir,
        #     dataset_name=self.dataset_name,
        #     split=self.split,
        #     split_tag=self.split_tag,
        # )

        df = load_parquet(f"{self.split}_{self.split_tag}_entities.parquet", self.data_dir / self.dataset_name)

        # --------------------------------------------------
        # Event node indexing
        # --------------------------------------------------
        add_node_index(
            df,
            id_col=self.node_id_col,
            index_col="event_idx",
        )

        data = HeteroData()

        # --------------------------------------------------
        # Event nodes
        # --------------------------------------------------
        data[self.EVENT].num_nodes = df["event_idx"].nunique()
        data[self.EVENT].y = torch.tensor(
            df[self.label_col].to_numpy(dtype="int64"),
            dtype=torch.long,
        )

        # --------------------------------------------------
        # Component indices (split-local, role-aware)
        # --------------------------------------------------
        # Sorted (not first-occurrence) order: index assignment must not
        # depend on row order upstream of this builder.
        actor1_index = {v: i for i, v in enumerate(sorted(df["Actor1ID"].dropna().unique()))}
        actor2_index = {v: i for i, v in enumerate(sorted(df["Actor2ID"].dropna().unique()))}
        geo_index = {v: i for i, v in enumerate(sorted(df["Event_GeoID"].dropna().unique()))}
        day_index = {v: i for i, v in enumerate(sorted(df["Day"].dropna().unique()))}

        data[self.ACTOR1].num_nodes = len(actor1_index)
        data[self.ACTOR2].num_nodes = len(actor2_index)
        data[self.GEO].num_nodes = len(geo_index)
        data[self.DAY].num_nodes = len(day_index)

        # --------------------------------------------------
        # Helper to attach bidirectional edges
        # --------------------------------------------------
        def attach_edges(
            edge_type,
            rev_edge_type,
            component_col: str,
            component_index: Dict,
        ):
            edge_fwd, edge_rev = build_event_component_edges(
                df=df,
                event_idx_col="event_idx",
                component_col=component_col,
                component_index=component_index,
            )
            data[edge_type].edge_index = edge_fwd
            data[rev_edge_type].edge_index = edge_rev

        # --------------------------------------------------
        # Edges
        # --------------------------------------------------
        attach_edges(self.HAS_ACTOR1, self.REV_HAS_ACTOR1, "Actor1ID", actor1_index)
        attach_edges(self.HAS_ACTOR2, self.REV_HAS_ACTOR2, "Actor2ID", actor2_index)
        attach_edges(self.OCCURRED_IN_GEO, self.REV_IN_GEO, "Event_GeoID", geo_index)
        attach_edges(self.OCCURRED_ON_DAY, self.REV_ON_DAY, "Day", day_index)

        return data
