# src/models/multi_domain/fusion.py
"""
Fusion mechanisms for the multi-domain geometric model.

All modules receive a list of per-domain view tensors (heterogeneous dims
are allowed) and output class logits directly.

Available types (set via model.fusion.type in YAML):
  concat_mlp       - concatenate views -> MLP  (geometry-blind)
  attention        - softmax attention over view projections -> MLP  (geometry-blind)
  gated            - sigmoid gates from global context -> gated sum -> MLP  (geometry-blind)
  geometry_aware   - log-map sphere views to tangent space first, then delegate
                     to any inner fusion type via model.fusion.inner_type

geometry_aware fusion
---------------------
Views whose encoders live on a hypersphere (HypersphericalEncoder,
RiemannianProductEncoder, RegionAwareEncoder, ProjectedEncoder) are mapped
from S^{d-1} to the tangent space at the north pole before fusion:

    log_p(x)[..., :-1]  :  S^{d-1} -> R^{d-1}   (drop the zero last coord)

Euclidean views (GNN actor encoder, EuclideanEncoder, MLP temporal encoders)
pass through unchanged.

The effective dimensions after mapping are passed to the inner fusion module,
so the MLP/attention/gated architecture is automatically sized correctly.
The manifold type per view is derived from the encoder configs in model.py
and never needs to be specified manually.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .riemannian import log_north


# =========================================================================
# Fusion implementations
# =========================================================================

class ConcatMLPFusion(nn.Module):
    """
    Late concatenation: cat(views) -> MLP -> logits.

    The simplest fusion strategy; each view contributes equally to the
    classifier input. Serves as the default baseline.
    """

    def __init__(
        self,
        view_dims: list[int],
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sum(view_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        return self.mlp(torch.cat(views, dim=-1))


class AttentionFusion(nn.Module):
    """
    Attention-weighted fusion over domain views.

    Each view is projected to attention_dim; a shared query vector scores
    each projected view via dot product. Softmax weights produce a weighted
    sum that is then classified by an MLP.

    This allows the model to dynamically emphasise whichever domain is most
    informative per event - e.g. geography for natural disasters, actors for
    diplomatic incidents.
    """

    def __init__(
        self,
        view_dims: list[int],
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        attention_dim: int = 64,
    ):
        super().__init__()
        # Per-view projection to attention space
        self.score_projs = nn.ModuleList([
            nn.Linear(d, attention_dim) for d in view_dims
        ])
        # Shared query for scoring
        self.query = nn.Parameter(torch.randn(attention_dim))
        # Per-view projection to value space
        self.value_projs = nn.ModuleList([
            nn.Linear(d, hidden_dim) for d in view_dims
        ])
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        # Attention scores: (B,) per view -> (B, num_views)
        scores = torch.stack([
            torch.tanh(proj(v)) @ self.query
            for proj, v in zip(self.score_projs, views)
        ], dim=-1)
        alpha = F.softmax(scores, dim=-1)  # (B, num_views)

        # Weighted sum of value projections
        values = torch.stack(
            [proj(v) for proj, v in zip(self.value_projs, views)], dim=1
        )  # (B, num_views, hidden_dim)
        z = (alpha.unsqueeze(-1) * values).sum(dim=1)  # (B, hidden_dim)
        return self.mlp(z)


class GatedFusion(nn.Module):
    """
    Gated fusion with per-domain sigmoid gates.

    A global context vector (concatenation of all views) produces one
    sigmoid gate per hidden unit via a linear layer. Each view is projected
    to hidden_dim, summed, then modulated element-wise by the gate before
    classification.

    This is a soft view-selection mechanism: the gate can suppress views
    with uninformative content (e.g. zero-filled coordinates) on a
    per-event basis.
    """

    def __init__(
        self,
        view_dims: list[int],
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        total_dim = sum(view_dims)
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.value_projs = nn.ModuleList([
            nn.Linear(d, hidden_dim) for d in view_dims
        ])
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        context = torch.cat(views, dim=-1)       # (B, total_dim)
        gate = self.gate_net(context)            # (B, hidden_dim)
        projected = sum(
            proj(v) for proj, v in zip(self.value_projs, views)
        )                                        # (B, hidden_dim)
        z = gate * projected
        return self.mlp(z)


class GeometryAwareFusion(nn.Module):
    """
    Geometry-aware fusion: maps each view to a common Euclidean tangent space
    before applying any inner fusion strategy.

    Sphere views  (S^{d-1})  ->  log_p(x)[..., :-1]  in R^{d-1}
    Euclidean views (R^d)    ->  identity             in R^d

    The inner fusion module (concat_mlp / attention / gated) is built with
    the *effective* dimensions after mapping, so it is correctly sized.

    Rationale
    ---------
    Directly concatenating a unit-norm sphere vector with an unconstrained
    Euclidean vector mixes incompatible scales and geometries.  The log map
    produces the geodesic displacement from the reference point (north pole),
    putting all views on equal geometric footing in a flat tangent space
    before the MLP or attention mechanism operates on them.
    """

    def __init__(
        self,
        view_manifolds: list[str],   # "sphere" | "euclidean" per view
        inner: nn.Module,            # pre-built inner fusion (sized to effective dims)
    ):
        super().__init__()
        self.view_manifolds = view_manifolds
        self.inner = inner

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        mapped = []
        for x, manifold in zip(views, self.view_manifolds):
            if manifold == "sphere":
                # Log map at north pole: S^{d-1} -> T_p ~= R^{d-1}
                mapped.append(log_north(x)[..., :-1])
            else:
                mapped.append(x)
        return self.inner(mapped)


# =========================================================================
# Factory
# =========================================================================

_BLIND_REGISTRY: dict[str, type] = {
    "concat_mlp": ConcatMLPFusion,
    "attention":  AttentionFusion,
    "gated":      GatedFusion,
}


def _build_blind_fusion(
    cfg: dict,
    view_dims: list[int],
    num_classes: int,
) -> nn.Module:
    """Build a geometry-blind fusion module from cfg and pre-computed view_dims."""
    fusion_type = cfg.get("type", "concat_mlp")
    if fusion_type not in _BLIND_REGISTRY:
        raise ValueError(
            f"Unknown inner fusion type '{fusion_type}'. "
            f"Available: {sorted(_BLIND_REGISTRY)}"
        )
    kwargs = dict(
        view_dims=view_dims,
        num_classes=num_classes,
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.2),
    )
    if fusion_type == "attention":
        kwargs["attention_dim"] = cfg.get("attention_dim", 64)
    return _BLIND_REGISTRY[fusion_type](**kwargs)


def build_fusion(
    cfg: dict,
    view_dims: list[int],
    num_classes: int,
    view_manifolds: list[str] | None = None,
) -> nn.Module:
    """
    Instantiate a fusion module from a config dict.

    Parameters
    ----------
    cfg : fusion sub-config (model.fusion in YAML)
    view_dims : output dimensions of [actor_encoder, geo_encoder, temporal_encoder]
    num_classes : number of target classes
    view_manifolds : manifold tag per view - "sphere" or "euclidean".
        Derived automatically from encoder types in MultiDomainGeometricModel.
        Only used when fusion type is "geometry_aware".
    """
    if view_manifolds is None:
        view_manifolds = ["euclidean"] * len(view_dims)

    fusion_type = cfg.get("type", "concat_mlp")

    if fusion_type == "geometry_aware":
        # Effective dims: sphere views lose one dimension (last coord is always 0)
        effective_dims = [
            d - 1 if m == "sphere" else d
            for d, m in zip(view_dims, view_manifolds)
        ]
        inner_cfg = {**cfg, "type": cfg.get("inner_type", "concat_mlp")}
        inner = _build_blind_fusion(inner_cfg, effective_dims, num_classes)
        return GeometryAwareFusion(view_manifolds, inner)

    all_types = set(_BLIND_REGISTRY) | {"geometry_aware"}
    if fusion_type not in _BLIND_REGISTRY:
        raise ValueError(
            f"Unknown fusion type '{fusion_type}'. "
            f"Available: {sorted(all_types)}"
        )
    return _build_blind_fusion(cfg, view_dims, num_classes)
