# src/models/multiview/encoders.py
"""
Component-specific geometric encoders for the multiview event model.

Each encoder is responsible for one event component and is designed to
match the mathematical domain of that component:

  ActorGNNEncoder          — WHO/WHOM   → relational graph  (permutation-equivariant)
  HypersphericalEncoder    — WHERE      → hypersphere S^{d-1} (geodesic geometry)
  TemporalProductEncoder   — WHEN       → ℝ × S¹ × S¹        (product manifold)

References (from the qualification exam, Chapter 4):
  - Relational: GNN with SAGEConv or GATConv
  - Spatial: geodetic-cartesian → S² → learnable projection → L2-normalised
  - Temporal: ℝ (linear progression) + S¹ (annual cycle) + S¹ (weekly cycle)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


# =========================================================================
# WHO/WHOM — Actor Graph GNN Encoder
# =========================================================================
class ActorGNNEncoder(nn.Module):
    """
    Encodes actor relational structure via message-passing on a co-occurrence graph.

    Node features (label-encoded actor attributes) are embedded, then refined
    through GNN layers that propagate information across actor neighbourhoods.
    For each event, Actor1 and Actor2 embeddings are retrieved and fused.

    Parameters
    ----------
    cardinalities : per-feature cardinality (including slot 0 for unknown)
    feat_embed_dim : embedding dimension per categorical actor attribute
    hidden_dim : GNN hidden/output dimension
    out_dim : final per-event representation dimension (after pair fusion)
    num_layers : number of GNN message-passing layers
    conv_type : 'sage' (GraphSAGE) or 'gat' (Graph Attention)
    dropout : dropout probability in fusion MLP
    """

    def __init__(
        self,
        cardinalities: list[int],
        feat_embed_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        conv_type: str = "sage",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv_type = conv_type.lower()
        self.dropout = dropout

        # One embedding table per actor attribute column; padding_idx=0 → zero vec for unknown
        self.feat_embeddings = nn.ModuleList([
            nn.Embedding(card, feat_embed_dim, padding_idx=0)
            for card in cardinalities
        ])
        node_input_dim = len(cardinalities) * feat_embed_dim

        # Project concatenated embeddings into GNN hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # GNN layers
        self.convs = nn.ModuleList([
            self._make_conv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Pair fusion: concat(actor1_emb, actor2_emb) → out_dim
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    # ------------------------------------------------------------------
    def _make_conv(self, in_dim: int, out_dim: int):
        if self.conv_type == "sage":
            return SAGEConv(in_dim, out_dim)
        if self.conv_type == "gat":
            return GATConv(in_dim, out_dim, heads=1, concat=False, dropout=self.dropout)
        raise ValueError(f"Unknown conv_type '{self.conv_type}'. Choose 'sage' or 'gat'.")

    def _encode_node_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, num_attrs) int64  →  (N, num_attrs * feat_embed_dim) float32"""
        parts = [emb(x[:, i]) for i, emb in enumerate(self.feat_embeddings)]
        return torch.cat(parts, dim=-1)

    def _run_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings for the full actor graph."""
        h = self.input_proj(self._encode_node_features(x))
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # (N_actors, hidden_dim)

    def forward(
        self,
        graph_x: torch.Tensor,       # (N_actors, 8) int64 — actor node features
        graph_edge_index: torch.Tensor,  # (2, E) int64 — co-occurrence edges
        actor1_idx: torch.Tensor,    # (B,) int64 — Actor1 node indices per event
        actor2_idx: torch.Tensor,    # (B,) int64 — Actor2 node indices per event
    ) -> torch.Tensor:               # (B, out_dim)
        actor_emb = self._run_gnn(graph_x, graph_edge_index)
        a1 = actor_emb[actor1_idx]   # (B, hidden_dim)
        a2 = actor_emb[actor2_idx]   # (B, hidden_dim)
        return self.pair_proj(torch.cat([a1, a2], dim=-1))  # (B, out_dim)


# =========================================================================
# WHERE — Hyperspherical Spatial Encoder
# =========================================================================
class HypersphericalEncoder(nn.Module):
    """
    Encodes geographic coordinates on a hypersphere (S^{out_dim - 1}).

    Input (lat, lon) pairs are first embedded on S² ⊂ ℝ³ (the natural
    geometry of the Earth's surface), then projected to a higher-dimensional
    hypersphere via a learnable MLP.  L2 normalisation ensures the output
    lives on the unit hypersphere, preserving geodesic structure.

    Advantages over Euclidean lat/lon encoding:
      - Distances correspond to true great-circle separation
      - Rotationally equivariant (no privileged orientation)
      - Consistent global geometry for events across all locations

    Parameters
    ----------
    out_dim : dimension of the output hyperspherical embedding
    hidden_dim : MLP hidden dimension
    """

    def __init__(self, out_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, lat_lon: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lat_lon : (B, 2) float32  — [latitude, longitude] in degrees

        Returns
        -------
        (B, out_dim) float32  — unit-norm embeddings on S^{out_dim - 1}
        """
        lat_rad = lat_lon[:, 0] * (torch.pi / 180.0)
        lon_rad = lat_lon[:, 1] * (torch.pi / 180.0)

        # Geodetic-Cartesian projection to S² ⊂ ℝ³
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        s2_coords = torch.stack([x, y, z], dim=-1)  # (B, 3)

        emb = self.proj(s2_coords)
        return F.normalize(emb, p=2, dim=-1)  # project to S^{out_dim - 1}


# =========================================================================
# WHEN — Temporal Product Manifold Encoder  (ℝ × S¹ × S¹)
# =========================================================================
class TemporalProductEncoder(nn.Module):
    """
    Encodes temporal information on the product manifold ℝ × S¹ × S¹.

    The three components capture distinct temporal scales:
      ℝ   — absolute linear progression (long-term trend)
      S¹  — annual periodicity  (day-of-year / 365, seasonal patterns)
      S¹  — weekly periodicity  (day-of-week / 7,   news cycle patterns)

    The fixed geometric encoding (5-dimensional) is followed by a learnable
    projection, allowing the model to weight and combine the temporal scales.

    Pre-computed temporal features expected as input (see MultiviewEventDataset):
      [0] t_linear  : normalised linear day index
      [1] sin_year  : sin(2π · doy / 365)
      [2] cos_year  : cos(2π · doy / 365)
      [3] sin_week  : sin(2π · dow / 7)
      [4] cos_week  : cos(2π · dow / 7)

    Parameters
    ----------
    out_dim : output embedding dimension
    hidden_dim : MLP hidden dimension
    """

    INPUT_DIM: int = 5  # fixed by the product manifold decomposition

    def __init__(self, out_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        time_features : (B, 5) float32 — pre-computed product manifold features

        Returns
        -------
        (B, out_dim) float32
        """
        return self.proj(time_features)
