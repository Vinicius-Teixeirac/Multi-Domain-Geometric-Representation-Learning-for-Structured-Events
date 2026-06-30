# src/models/multi_domain/actor_encoders.py
"""
Actor domain encoders for the multi-domain geometric model.

Available types (set via model.actor.type in YAML):
  sage_gnn       - GraphSAGE on unweighted co-occurrence graph (transductive)
  gat_gnn        - Multi-head Graph Attention Network (transductive)
  weighted_gnn   - GCN with normalised co-occurrence edge weights (transductive)
  attribute_only - Attribute embedding only, no message passing (inductive-ready)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


# =========================================================================
# Shared base
# =========================================================================

class _ActorEncoderBase(nn.Module):
    """Shared categorical node feature embedding for all actor encoders."""

    def __init__(self, cardinalities: list[int], feat_embed_dim: int):
        """
        Parameters
        ----------
        cardinalities : list[int]
            Per-attribute cardinality (including the unknown slot at index 0).
        feat_embed_dim : int
            Embedding dimension per categorical actor attribute.
        """
        super().__init__()
        self.feat_embeddings = nn.ModuleList([
            nn.Embedding(card, feat_embed_dim, padding_idx=0)
            for card in cardinalities
        ])

    def _encode_node_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, num_attrs) int64 -> (N, num_attrs * feat_embed_dim) float32"""
        return torch.cat(
            [emb(x[:, i]) for i, emb in enumerate(self.feat_embeddings)], dim=-1
        )


# =========================================================================
# Encoder implementations
# =========================================================================

class ActorSAGEEncoder(_ActorEncoderBase):
    """
    GraphSAGE actor encoder on an unweighted co-occurrence graph.

    Neighbourhood aggregation treats all co-occurring actors equally;
    only structural topology influences the resulting embeddings.

    Parameters
    ----------
    cardinalities : list[int]
        Per-attribute cardinality including the unknown slot at index 0.
    feat_embed_dim : int
        Embedding dimension per categorical actor attribute.
    hidden_dim : int
        Hidden dimension for GNN layers and the input projection.
    out_dim : int
        Output dimension of the pair embedding.
    num_layers : int
        Number of SAGEConv message-passing layers.
    dropout : float
        Dropout probability applied after each GNN layer.
    """

    def __init__(
        self,
        cardinalities: list[int],
        feat_embed_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(cardinalities, feat_embed_dim)
        node_input_dim = len(cardinalities) * feat_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.dropout = dropout

    def forward(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        actor1_idx: torch.Tensor,
        actor2_idx: torch.Tensor,
        graph_edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run SAGE message passing and return the concatenated actor-pair embedding."""
        h = self.input_proj(self._encode_node_features(graph_x))
        for conv in self.convs:
            h = F.relu(conv(h, graph_edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        a1, a2 = h[actor1_idx], h[actor2_idx]
        return self.pair_proj(torch.cat([a1, a2], dim=-1))


class ActorGATEncoder(_ActorEncoderBase):
    """
    Multi-head Graph Attention actor encoder.

    Attention coefficients weight each neighbour's contribution, allowing
    the model to focus on the most relevant co-occurring actors per event.

    Uses concat=False so output dim = hidden_dim regardless of gat_heads,
    keeping pair_proj input dimension stable across head counts.

    Parameters
    ----------
    cardinalities : list[int]
        Per-attribute cardinality including the unknown slot at index 0.
    feat_embed_dim : int
        Embedding dimension per categorical actor attribute.
    hidden_dim : int
        Hidden dimension for GNN layers and the input projection.
    out_dim : int
        Output dimension of the pair embedding.
    num_layers : int
        Number of GATConv message-passing layers.
    dropout : float
        Dropout probability applied after each GNN layer.
    gat_heads : int
        Number of attention heads per GATConv layer.
    """

    def __init__(
        self,
        cardinalities: list[int],
        feat_embed_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        gat_heads: int = 4,
    ):
        super().__init__(cardinalities, feat_embed_dim)
        node_input_dim = len(cardinalities) * feat_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # concat=False: each head outputs hidden_dim; heads are averaged -> (N, hidden_dim)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=False, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.dropout = dropout

    def forward(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        actor1_idx: torch.Tensor,
        actor2_idx: torch.Tensor,
        graph_edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run GAT message passing and return the concatenated actor-pair embedding."""
        h = self.input_proj(self._encode_node_features(graph_x))
        for conv in self.convs:
            h = F.relu(conv(h, graph_edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        a1, a2 = h[actor1_idx], h[actor2_idx]
        return self.pair_proj(torch.cat([a1, a2], dim=-1))


class ActorWeightedEncoder(_ActorEncoderBase):
    """
    GCN actor encoder using normalised co-occurrence edge weights.

    Edge weights are the normalised event co-occurrence counts built by
    actor_graph_builder (stored in graph.edge_attr). GCNConv uses these
    during symmetric normalisation, giving more influence to actor pairs
    that co-occur frequently in training events.

    Parameters
    ----------
    cardinalities : list[int]
        Per-attribute cardinality including the unknown slot at index 0.
    feat_embed_dim : int
        Embedding dimension per categorical actor attribute.
    hidden_dim : int
        Hidden dimension for GNN layers and the input projection.
    out_dim : int
        Output dimension of the pair embedding.
    num_layers : int
        Number of GCNConv message-passing layers.
    dropout : float
        Dropout probability applied after each GNN layer.
    """

    def __init__(
        self,
        cardinalities: list[int],
        feat_embed_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(cardinalities, feat_embed_dim)
        node_input_dim = len(cardinalities) * feat_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.dropout = dropout

    def forward(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        actor1_idx: torch.Tensor,
        actor2_idx: torch.Tensor,
        graph_edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run weighted GCN message passing and return the concatenated actor-pair embedding."""
        # GCNConv expects edge_weight as 1-D (E,) float tensor; None -> unweighted fallback
        edge_weight = (
            graph_edge_attr
            if graph_edge_attr is not None and graph_edge_attr.numel() > 0
            else None
        )
        h = self.input_proj(self._encode_node_features(graph_x))
        for conv in self.convs:
            h = F.relu(conv(h, graph_edge_index, edge_weight=edge_weight))
            h = F.dropout(h, p=self.dropout, training=self.training)
        a1, a2 = h[actor1_idx], h[actor2_idx]
        return self.pair_proj(torch.cat([a1, a2], dim=-1))


class ActorAttributeEncoder(_ActorEncoderBase):
    """
    Inductive actor encoder: categorical attribute embeddings only.

    No message passing is performed. Actor embeddings are derived purely
    from the 8 categorical attribute columns, making this approach suitable
    for unseen actors at inference time (fully inductive setting).

    Parameters
    ----------
    cardinalities : list[int]
        Per-attribute cardinality including the unknown slot at index 0.
    feat_embed_dim : int
        Embedding dimension per categorical actor attribute.
    hidden_dim : int
        Hidden dimension of the attribute projection MLP.
    out_dim : int
        Output dimension of the pair embedding.
    dropout : float
        Dropout probability applied inside the projection MLP.
    """

    def __init__(
        self,
        cardinalities: list[int],
        feat_embed_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__(cardinalities, feat_embed_dim)
        node_input_dim = len(cardinalities) * feat_embed_dim
        self.attr_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        actor1_idx: torch.Tensor,
        actor2_idx: torch.Tensor,
        graph_edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed actors from attributes only (no message passing) and return the pair embedding."""
        h = self.attr_proj(self._encode_node_features(graph_x))
        a1, a2 = h[actor1_idx], h[actor2_idx]
        return self.pair_proj(torch.cat([a1, a2], dim=-1))


# =========================================================================
# Factory
# =========================================================================

_ACTOR_REGISTRY: dict[str, type] = {
    "sage_gnn":      ActorSAGEEncoder,
    "gat_gnn":       ActorGATEncoder,
    "weighted_gnn":  ActorWeightedEncoder,
    "attribute_only": ActorAttributeEncoder,
}


def build_actor_encoder(cfg: dict, cardinalities: list[int]) -> nn.Module:
    """
    Instantiate an actor encoder from a config dict.

    Parameters
    ----------
    cfg : actor sub-config (model.actor in YAML)
    cardinalities : per-attribute cardinalities from actor_graph_builder
    """
    encoder_type = cfg.get("type", "sage_gnn")
    if encoder_type not in _ACTOR_REGISTRY:
        raise ValueError(
            f"Unknown actor encoder type '{encoder_type}'. "
            f"Available: {sorted(_ACTOR_REGISTRY)}"
        )
    kwargs = dict(
        cardinalities=cardinalities,
        feat_embed_dim=cfg.get("feat_embed_dim", 16),
        hidden_dim=cfg.get("hidden_dim", 128),
        out_dim=cfg.get("out_dim", 64),
        dropout=cfg.get("dropout", 0.2),
    )
    if encoder_type in ("sage_gnn", "gat_gnn", "weighted_gnn"):
        kwargs["num_layers"] = cfg.get("num_layers", 2)
    if encoder_type == "gat_gnn":
        kwargs["gat_heads"] = cfg.get("gat_heads", 4)
    return _ACTOR_REGISTRY[encoder_type](**kwargs)
