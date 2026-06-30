# src/models/multi_domain/geo_encoders.py
"""
Geospatial domain encoders for the multi-domain geometric model.

Available types (set via model.geo.type in YAML):
  hyperspherical - Truly Riemannian: every representation lives on a hypersphere.
                   Operations use log/exp maps; no Euclidean excursions.
  projected      - S^2 -> Euclidean MLP -> L2-normalise (ablation baseline).
  euclidean      - Direct MLP on (lat, lon) with no sphere inductive bias.
  region_aware   - Riemannian spherical path blended with country embedding.

Riemannian primitives
---------------------
All spherical encoders use the north-pole (p = (0,...,0,1)) as the reference
point.  Choosing the north pole simplifies the log/exp maps substantially:

  log_p(x) = arccos(x_{-1}) * [x_{:-1} / ||x_{:-1}||, 0]
  exp_p(v) = [sin(||v||)/||v|| * v_{:-1}, cos(||v||)]

The last coordinate of tangent vectors at p is identically 0 by construction,
so SphericalLinear operates on the (d-1)-dimensional tangent sub-coordinates
and SphereReLU preserves the tangent constraint (ReLU(0) = 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .riemannian import SphericalLinear, SphereReLU, log_north, exp_north

_DEG_TO_RAD: float = torch.pi / 180.0  # conversion factor for geodetic coordinates


# =========================================================================
# S^2 helper
# =========================================================================

def _to_s2(lat_lon: torch.Tensor) -> torch.Tensor:
    """Convert (lat, lon) degrees -> unit S^2 coordinates in R^3."""
    lat = lat_lon[:, 0] * _DEG_TO_RAD
    lon = lat_lon[:, 1] * _DEG_TO_RAD
    return torch.stack([
        torch.cos(lat) * torch.cos(lon),
        torch.cos(lat) * torch.sin(lon),
        torch.sin(lat),
    ], dim=-1)  # (B, 3)


# =========================================================================
# Encoder implementations
# =========================================================================

class HypersphericalEncoder(nn.Module):
    """
    Truly hyperspherical encoder - every intermediate and final representation
    lives on a hypersphere.  No Euclidean excursions.

    Architecture (all steps stay on manifolds)
    ------------------------------------------
    (lat, lon)  -> _to_s2 ->  S^2 subset of R^3
                -> SphericalLinear(3 -> hidden_dim)  ->  S^{hidden-1}
                -> SphereReLU                        ->  S^{hidden-1}
                -> SphericalLinear(hidden_dim -> out_dim)  ->  S^{out-1}

    Inductive bias
    --------------
    Geographic proximity is reflected as angular proximity on every
    intermediate sphere.  Gradients flow through geodesics (log/exp maps)
    rather than through arbitrary Euclidean paths, giving the network a
    principled geometric inductive bias for spatial event representation.
    """

    def __init__(self, out_dim: int = 32, hidden_dim: int = 64):
        """
        Parameters
        ----------
        out_dim : int
            Output embedding dimension; output lives on S^{out_dim-1}.
        hidden_dim : int
            Intermediate hypersphere dimension between the two SphericalLinear layers.
        """
        super().__init__()
        self.layer1 = SphericalLinear(3, hidden_dim)
        self.act    = SphereReLU()
        self.layer2 = SphericalLinear(hidden_dim, out_dim)

    def forward(
        self,
        lat_lon: torch.Tensor,
        geo_country_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode (lat, lon) coordinates to a unit vector on S^{out_dim-1}.

        Parameters
        ----------
        lat_lon : torch.Tensor of shape (B, 2)
            Latitude and longitude in degrees.
        geo_country_idx : ignored (accepted for API uniformity)

        Returns
        -------
        torch.Tensor of shape (B, out_dim) on S^{out_dim-1}
        """
        x = _to_s2(lat_lon)   # (B, 3)          on S^2
        x = self.layer1(x)    # (B, hidden_dim)  on S^{hidden-1}
        x = self.act(x)       # (B, hidden_dim)  on S^{hidden-1}
        x = self.layer2(x)    # (B, out_dim)     on S^{out-1}
        return x


class ProjectedEncoder(nn.Module):
    """
    Ablation baseline: S^2 -> Euclidean MLP -> L2-normalise.

    The output lands on S^{out_dim-1} but intermediate activations are in
    unconstrained Euclidean space.  Use type: projected to measure the value
    of the Riemannian inductive bias introduced by HypersphericalEncoder.
    """

    def __init__(self, out_dim: int = 32, hidden_dim: int = 64):
        """
        Parameters
        ----------
        out_dim : int
            Output embedding dimension; L2-normalised to unit norm.
        hidden_dim : int
            Hidden width of the MLP operating in Euclidean space.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        lat_lon: torch.Tensor,
        geo_country_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Project S^2 coordinates through a Euclidean MLP and L2-normalise the result.

        Parameters
        ----------
        lat_lon : torch.Tensor of shape (B, 2)
        geo_country_idx : ignored

        Returns
        -------
        torch.Tensor of shape (B, out_dim), unit-norm.
        """
        return F.normalize(self.proj(_to_s2(lat_lon)), p=2, dim=-1)


class EuclideanEncoder(nn.Module):
    """
    Direct MLP on (lat, lon) - no sphere inductive bias whatsoever.
    Use type: euclidean to ablate whether any sphere geometry helps.
    """

    def __init__(self, out_dim: int = 32, hidden_dim: int = 64):
        """
        Parameters
        ----------
        out_dim : int
            Output embedding dimension.
        hidden_dim : int
            Hidden width of the MLP.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        lat_lon: torch.Tensor,
        geo_country_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply MLP directly to (lat, lon) without any spherical pre-processing.

        Parameters
        ----------
        lat_lon : torch.Tensor of shape (B, 2)
        geo_country_idx : ignored

        Returns
        -------
        torch.Tensor of shape (B, out_dim)
        """
        return self.proj(lat_lon)


class RegionAwareEncoder(nn.Module):
    """
    Riemannian spherical path blended with a learnable country embedding.

    Architecture
    ------------
    (lat, lon)  -> S^2
                -> SphericalLinear(3 -> hidden_dim)  ->  S^{hidden-1}
                -> SphereReLU                        ->  S^{hidden-1}
                -> log_north -> tangent coords         in R^{hidden-1}
                -> concat with region embedding       in R^{hidden-1 + r}
                -> Linear -> R^{out-1}
                -> exp_north                          ->  S^{out-1}

    The region embedding augments the purely geometric signal with geopolitical
    context (country-level event patterns).  It is injected in the tangent space
    of the intermediate sphere, then the result is mapped back to S^{out-1}.
    """

    def __init__(
        self,
        out_dim: int = 32,
        hidden_dim: int = 64,
        region_cardinality: int = 1,
        region_embed_dim: int = 16,
    ):
        """
        Parameters
        ----------
        out_dim : int
            Output embedding dimension; lives on S^{out_dim-1}.
        hidden_dim : int
            Intermediate hypersphere dimension.
        region_cardinality : int
            Number of unique country codes + 1 (index 0 = unknown).
        region_embed_dim : int
            Dimension of the learnable country embedding.
        """
        super().__init__()
        # padding_idx=0 -> zero vector for unknown/missing country
        self.region_embed = nn.Embedding(
            region_cardinality, region_embed_dim, padding_idx=0
        )
        self.sphere_layer = SphericalLinear(3, hidden_dim)
        self.sphere_act   = SphereReLU()
        # Tangent coords are (hidden-1)-dimensional; blend with region embed
        self.blend = nn.Linear(hidden_dim - 1 + region_embed_dim, out_dim - 1)

    def forward(
        self,
        lat_lon: torch.Tensor,
        geo_country_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Blend spherical geo path with a country embedding and return a point on S^{out-1}.

        Parameters
        ----------
        lat_lon : torch.Tensor of shape (B, 2)
            Latitude and longitude in degrees.
        geo_country_idx : torch.Tensor of shape (B,) or None
            Integer country index; None uses the zero (unknown) embedding.

        Returns
        -------
        torch.Tensor of shape (B, out_dim) on S^{out_dim-1}
        """
        x = _to_s2(lat_lon)                                # (B, 3)         on S^2
        h = self.sphere_layer(x)                           # (B, hidden)    on S^{hidden-1}
        h = self.sphere_act(h)                             # (B, hidden)    on S^{hidden-1}
        v = log_north(h)[..., :-1]                        # (B, hidden-1)  tangent coords

        if geo_country_idx is None:
            r = torch.zeros(
                lat_lon.size(0), self.region_embed.embedding_dim,
                device=lat_lon.device, dtype=lat_lon.dtype,
            )
        else:
            r = self.region_embed(geo_country_idx.long())  # (B, region_embed_dim)

        combined = torch.cat([v, r], dim=-1)               # (B, hidden-1+r)
        u_tan    = self.blend(combined)                    # (B, out-1)
        u        = torch.cat([u_tan,
                              torch.zeros_like(u_tan[..., :1])], dim=-1)  # (B, out)
        return exp_north(u)                               # (B, out_dim)   on S^{out-1}


# =========================================================================
# Factory
# =========================================================================

_GEO_REGISTRY: dict[str, type] = {
    "hyperspherical": HypersphericalEncoder,
    "projected":      ProjectedEncoder,
    "euclidean":      EuclideanEncoder,
    "region_aware":   RegionAwareEncoder,
}


def build_geo_encoder(cfg: dict, region_cardinality: int = 1) -> nn.Module:
    """
    Instantiate a geo encoder from a config dict.

    Parameters
    ----------
    cfg               : model.geo sub-config (from YAML)
    region_cardinality: total country codes + 1 for unknown.
                        Only used when type == 'region_aware'.
    """
    encoder_type = cfg.get("type", "hyperspherical")
    if encoder_type not in _GEO_REGISTRY:
        raise ValueError(
            f"Unknown geo encoder type '{encoder_type}'. "
            f"Available: {sorted(_GEO_REGISTRY)}"
        )
    kwargs = dict(
        out_dim=cfg.get("out_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 64),
    )
    if encoder_type == "region_aware":
        kwargs["region_cardinality"] = region_cardinality
        kwargs["region_embed_dim"]   = cfg.get("region_embed_dim", 16)
    return _GEO_REGISTRY[encoder_type](**kwargs)
