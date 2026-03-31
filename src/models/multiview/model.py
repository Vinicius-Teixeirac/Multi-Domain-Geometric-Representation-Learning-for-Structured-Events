# src/models/multiview/model.py
"""
MultiviewGeometricModel

Implements the composite event representation framework proposed in Chapter 4
of the qualification exam:

    φ(e) = Ψ(φ_who, φ_when, φ_where)

where each component encoder φ_• maps its input to a geometrically appropriate
latent space, and Ψ is a late-fusion operator (concatenation + MLP classifier).

Component geometries
--------------------
  WHO/WHOM  → graph relational space     (permutation-equivariant GNN)
  WHERE     → hypersphere S^{d-1}        (geodesic spatial geometry)
  WHEN      → product manifold ℝ×S¹×S¹  (linear + annual + weekly cycles)

The actor co-occurrence graph is stored as registered PyTorch buffers so that
model.to(device) automatically moves it to the correct device.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import ActorGNNEncoder, HypersphericalEncoder, TemporalProductEncoder


class MultiviewGeometricModel(nn.Module):
    """
    Multiview geometric event classifier.

    Parameters
    ----------
    actor_cardinalities : list[int]
        Per-attribute cardinality for the actor node feature embeddings.
        Obtained from ``build_actor_graph`` in actor_graph_builder.py.
    num_classes : int
        Number of target classes (4 for QuadClass).
    actor_feat_embed_dim : int
        Embedding dimension per actor categorical attribute.
    actor_hidden_dim : int
        GNN hidden dimension (also used for actor node embeddings after GNN).
    actor_out_dim : int
        Dimension of the relational (WHO) view embedding.
    geo_hidden_dim : int
        Hidden dimension for the hyperspherical projection MLP.
    geo_out_dim : int
        Dimension of the spatial (WHERE) view embedding.
    time_hidden_dim : int
        Hidden dimension for the temporal projection MLP.
    time_out_dim : int
        Dimension of the temporal (WHEN) view embedding.
    fusion_hidden_dim : int
        Hidden dimension for the late-fusion MLP classifier.
    num_gnn_layers : int
        Number of message-passing layers in the actor GNN.
    conv_type : str
        GNN convolution type: 'sage' or 'gat'.
    dropout : float
        Dropout probability applied in fusion MLP and actor encoder.
    """

    def __init__(
        self,
        actor_cardinalities: list[int],
        num_classes: int = 4,
        actor_feat_embed_dim: int = 16,
        actor_hidden_dim: int = 128,
        actor_out_dim: int = 64,
        geo_hidden_dim: int = 64,
        geo_out_dim: int = 32,
        time_hidden_dim: int = 32,
        time_out_dim: int = 16,
        fusion_hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        conv_type: str = "sage",
        dropout: float = 0.2,
    ):
        super().__init__()

        # --- Component encoders ---
        self.actor_encoder = ActorGNNEncoder(
            cardinalities=actor_cardinalities,
            feat_embed_dim=actor_feat_embed_dim,
            hidden_dim=actor_hidden_dim,
            out_dim=actor_out_dim,
            num_layers=num_gnn_layers,
            conv_type=conv_type,
            dropout=dropout,
        )
        self.geo_encoder = HypersphericalEncoder(
            out_dim=geo_out_dim,
            hidden_dim=geo_hidden_dim,
        )
        self.time_encoder = TemporalProductEncoder(
            out_dim=time_out_dim,
            hidden_dim=time_hidden_dim,
        )

        # --- Late-fusion classifier Ψ ---
        fused_dim = actor_out_dim + geo_out_dim + time_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        # --- Actor graph (registered as buffers → moved by model.to(device)) ---
        # Placeholder shapes; replaced by set_actor_graph() before training.
        num_attrs = len(actor_cardinalities)
        self.register_buffer(
            "_graph_x",
            torch.zeros(1, num_attrs, dtype=torch.long),
        )
        self.register_buffer(
            "_graph_edge_index",
            torch.zeros(2, 0, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------
    def set_actor_graph(
        self, graph_x: torch.Tensor, graph_edge_index: torch.Tensor
    ) -> None:
        """
        Load the pre-built actor co-occurrence graph.

        Must be called before the first forward pass (typically in the runner
        after DataModule.setup()). After this call, model.to(device) will
        correctly move the graph to the target device via the registered buffers.
        """
        self.register_buffer("_graph_x", graph_x)
        self.register_buffer("_graph_edge_index", graph_edge_index)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        actor1_idx: torch.Tensor,     # (B,) int64
        actor2_idx: torch.Tensor,     # (B,) int64
        geo: torch.Tensor,            # (B, 2) float32  [lat, lon] degrees
        time_features: torch.Tensor,  # (B, 5) float32  product-manifold features
    ) -> torch.Tensor:                # (B, num_classes)
        z_rel  = self.actor_encoder(self._graph_x, self._graph_edge_index, actor1_idx, actor2_idx)
        z_geo  = self.geo_encoder(geo)
        z_time = self.time_encoder(time_features)

        z = torch.cat([z_rel, z_geo, z_time], dim=-1)
        return self.classifier(z)

    def forward_batch(self, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interface expected by src/training/train.py.

        Parameters
        ----------
        batch : dict with keys:
            'actor1_idx'    (B,)    int64
            'actor2_idx'    (B,)    int64
            'geo'           (B, 2)  float32
            'time_features' (B, 5)  float32
            'labels'        (B,)    int64
        device : str — target device string ('cuda', 'cpu', etc.)

        Returns
        -------
        logits  : (B, num_classes)
        targets : (B,)
        """
        actor1_idx    = batch["actor1_idx"].to(device)
        actor2_idx    = batch["actor2_idx"].to(device)
        geo           = batch["geo"].to(device)
        time_features = batch["time_features"].to(device)
        labels        = batch["labels"].to(device)

        logits = self.forward(actor1_idx, actor2_idx, geo, time_features)
        return logits, labels
