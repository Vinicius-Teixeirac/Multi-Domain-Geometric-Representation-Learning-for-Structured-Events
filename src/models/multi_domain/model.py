# src/models/multi_domain/model.py
"""
MultiDomainGeometricModel

Implements the composite event representation framework proposed in Chapter 4
of the qualification exam:

    phi(e) = Psi(phi_who, phi_when, phi_where)

All architectural choices - encoder type, hyperparameters, and fusion mechanism -
are driven by a YAML config dict, enabling systematic comparison of geometric
inductive biases without code changes.

Component domains
-----------------
  WHO/WHOM  -> actor encoder    (see actor_encoders.py)
  WHERE     -> geo encoder      (see geo_encoders.py)
  WHEN      -> temporal encoder (see temporal_encoders.py)

Fusion
------
  Psi         -> fusion mechanism (see fusion.py)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .actor_encoders import build_actor_encoder
from .fusion import build_fusion
from .geo_encoders import build_geo_encoder
from .temporal_encoders import build_temporal_encoder


class MultiDomainGeometricModel(nn.Module):
    """
    Configurable multi-domain geometric event classifier.

    Parameters
    ----------
    model_cfg : dict
        The model sub-section of the experiment YAML config. Must contain
        sub-keys 'actor', 'geo', 'temporal', and 'fusion', each specifying
        a 'type' and type-specific hyperparameters.
    actor_cardinalities : list[int]
        Per-attribute cardinality for actor node feature embeddings.
        Obtained from ``build_actor_graph`` in actor_graph_builder.py.
    geo_country_cardinality : int
        Number of unique country codes + 1 (unknown slot at index 0).
        Only used when geo.type == 'region_aware'.
    num_classes : int
        Number of target classes.
    """

    def __init__(
        self,
        model_cfg: dict,
        actor_cardinalities: list[int],
        geo_country_cardinality: int,
        num_classes: int,
    ):
        super().__init__()

        actor_cfg    = model_cfg["actor"]
        geo_cfg      = model_cfg["geo"]
        temporal_cfg = model_cfg["temporal"]
        fusion_cfg   = model_cfg["fusion"]

        self.actor_encoder    = build_actor_encoder(actor_cfg, actor_cardinalities)
        self.geo_encoder      = build_geo_encoder(geo_cfg, geo_country_cardinality)
        self.temporal_encoder = build_temporal_encoder(temporal_cfg)

        view_dims = [
            actor_cfg["out_dim"],
            geo_cfg["out_dim"],
            temporal_cfg["out_dim"],
        ]

        # Derive manifold type per view from encoder configs so that
        # GeometryAwareFusion can apply the log map only where appropriate.
        # Actor GNN output is always Euclidean; geo/temporal depend on type.
        _SPHERE_GEO_TYPES      = {"hyperspherical", "region_aware", "projected"}
        _SPHERE_TEMPORAL_TYPES = {"riemannian_product"}
        view_manifolds = [
            "euclidean",
            "sphere" if geo_cfg.get("type", "") in _SPHERE_GEO_TYPES      else "euclidean",
            "sphere" if temporal_cfg.get("type", "") in _SPHERE_TEMPORAL_TYPES else "euclidean",
        ]

        self.fusion = build_fusion(fusion_cfg, view_dims, num_classes, view_manifolds)

        # --- Actor graph (registered as buffers -> moved by model.to(device)) ---
        # Placeholders replaced by set_actor_graph() before training.
        num_attrs = len(actor_cardinalities)
        self.register_buffer("_graph_x",          torch.zeros(1, num_attrs, dtype=torch.long))
        self.register_buffer("_graph_edge_index",  torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("_graph_edge_attr",   torch.zeros(0, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------
    def set_actor_graph(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_edge_attr: torch.Tensor | None = None,
    ) -> None:
        """
        Load the pre-built actor co-occurrence graph as registered buffers.

        Must be called before the first forward pass (typically in the runner
        after DataModule.setup()). model.to(device) then moves all graph
        tensors to the correct device automatically.

        Parameters
        ----------
        graph_x : (N_actors, 8) int64 - node feature matrix
        graph_edge_index : (2, E) int64 - undirected co-occurrence edges
        graph_edge_attr : (E,) float32 - normalised edge weights (optional).
            Required by weighted_gnn; ignored by all other actor encoders.
        """
        self.register_buffer("_graph_x",         graph_x)
        self.register_buffer("_graph_edge_index", graph_edge_index)
        # Always register so device transfer works uniformly
        edge_attr = (
            graph_edge_attr
            if graph_edge_attr is not None
            else torch.zeros(graph_edge_index.shape[1], dtype=torch.float32)
        )
        self.register_buffer("_graph_edge_attr", edge_attr)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        actor1_idx:      torch.Tensor,              # (B,) int64
        actor2_idx:      torch.Tensor,              # (B,) int64
        geo:             torch.Tensor,              # (B, 2) float32  [lat, lon]
        time_features:   torch.Tensor,              # (B, 3) float32  [t_linear, doy, dow]
        geo_country_idx: torch.Tensor | None = None,  # (B,) int64
    ) -> torch.Tensor:                              # (B, num_classes)
        z_actor = self.actor_encoder(
            self._graph_x, self._graph_edge_index,
            actor1_idx, actor2_idx,
            graph_edge_attr=self._graph_edge_attr,
        )
        z_geo  = self.geo_encoder(geo, geo_country_idx=geo_country_idx)
        z_time = self.temporal_encoder(time_features)
        return self.fusion([z_actor, z_geo, z_time])

    def forward_batch(
        self, batch: dict, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interface expected by src/training/train.py.

        Parameters
        ----------
        batch : dict with keys:
            'actor1_idx'      (B,)    int64
            'actor2_idx'      (B,)    int64
            'geo'             (B, 2)  float32
            'time_features'   (B, 3)  float32  [t_linear, doy, dow]
            'geo_country_idx' (B,)    int64     optional
            'labels'          (B,)    int64
        device : target device string ('cuda', 'cpu', etc.)
        """
        actor1_idx    = batch["actor1_idx"].to(device)
        actor2_idx    = batch["actor2_idx"].to(device)
        geo           = batch["geo"].to(device)
        time_features = batch["time_features"].to(device)
        labels        = batch["labels"].to(device)

        geo_country_idx = batch.get("geo_country_idx")
        if geo_country_idx is not None:
            geo_country_idx = geo_country_idx.to(device)

        logits = self.forward(actor1_idx, actor2_idx, geo, time_features, geo_country_idx)
        return logits, labels
