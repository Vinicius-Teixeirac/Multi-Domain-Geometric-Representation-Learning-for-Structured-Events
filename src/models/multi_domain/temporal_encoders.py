# src/models/multi_domain/temporal_encoders.py
"""
Temporal domain encoders for the multi-domain geometric model.

All encoders receive a (B, 3) float32 tensor from the dataset:
  [0] t_linear  - normalised linear day index  (z-scored on training split)
  [1] doy       - day-of-year  (1 - 366, float)
  [2] dow       - day-of-week  (0 - 6,   float; Mon = 0)

Encoders are responsible for all further transformations (sin/cos, Fourier).
This decoupling lets each variant impose a different temporal geometry without
touching the dataset or DataLoader.

Available types (set via model.temporal.type in YAML):
  product_manifold    - fixed periods R x S^1(365) x S^1(7) + Euclidean MLP
  learnable_period    - same structure with learnable cycle periods
  fourier             - learnable multi-frequency Fourier feature map + MLP
  riemannian_product  - truly manifold-aware: S^1 components processed via
                        SphericalLinear; output lives on S^{out-1}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .riemannian import SphericalLinear, log_north, exp_north

_PERIOD_ANNUAL: float = 365.0  # calendar days per year
_PERIOD_WEEKLY: float = 7.0    # calendar days per week


# =========================================================================
# Encoder implementations
# =========================================================================

class ProductManifoldEncoder(nn.Module):
    """
    Fixed-period product manifold encoder: R x S^1(365) x S^1(7).

    Computes the 5-dim encoding:
      [t_linear, sin(2pi*doy/365), cos(2pi*doy/365), sin(2pi*dow/7), cos(2pi*dow/7)]
    then projects to out_dim via a learnable MLP.

    Behaviourally equivalent to the original TemporalProductEncoder;
    sin/cos computation is simply moved from the dataset to here.
    """

    _ENCODED_DIM: int = 5

    def __init__(self, out_dim: int = 16, hidden_dim: int = 32):
        """
        Parameters
        ----------
        out_dim : int
            Output embedding dimension.
        hidden_dim : int
            Hidden width of the MLP that maps the 5-dim encoding to out_dim.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self._ENCODED_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Encode time features using fixed annual/weekly periods.

        Parameters
        ----------
        time_features : torch.Tensor of shape (B, 3)
            Columns: [t_linear, doy, dow].

        Returns
        -------
        torch.Tensor of shape (B, out_dim)
        """
        t   = time_features[:, 0:1]
        doy = time_features[:, 1:2]
        dow = time_features[:, 2:3]
        encoded = torch.cat([
            t,
            torch.sin(2 * torch.pi * doy / _PERIOD_ANNUAL),
            torch.cos(2 * torch.pi * doy / _PERIOD_ANNUAL),
            torch.sin(2 * torch.pi * dow / _PERIOD_WEEKLY),
            torch.cos(2 * torch.pi * dow / _PERIOD_WEEKLY),
        ], dim=-1)
        return self.proj(encoded)


class LearnablePeriodEncoder(nn.Module):
    """
    Product manifold encoder with learnable cycle periods.

    Annual and weekly periods are initialised to 365.0 and 7.0 but are
    free to adapt to the data. Log-space parameterisation keeps periods
    strictly positive throughout training.

    This lets the model discover dominant periodicities that differ from
    calendar cycles - e.g. political election cycles, seasonal conflict
    patterns specific to the GDELT event distribution.
    """

    _ENCODED_DIM: int = 5

    def __init__(self, out_dim: int = 16, hidden_dim: int = 32):
        """
        Parameters
        ----------
        out_dim : int
            Output embedding dimension.
        hidden_dim : int
            Hidden width of the projection MLP.
        """
        super().__init__()
        self.log_period_annual = nn.Parameter(torch.log(torch.tensor(_PERIOD_ANNUAL)))
        self.log_period_weekly = nn.Parameter(torch.log(torch.tensor(_PERIOD_WEEKLY)))
        self.proj = nn.Sequential(
            nn.Linear(self._ENCODED_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Encode time features using learnable cycle periods.

        Parameters
        ----------
        time_features : torch.Tensor of shape (B, 3)
            Columns: [t_linear, doy, dow].

        Returns
        -------
        torch.Tensor of shape (B, out_dim)
        """
        t   = time_features[:, 0:1]
        doy = time_features[:, 1:2]
        dow = time_features[:, 2:3]
        T_year = self.log_period_annual.exp()
        T_week = self.log_period_weekly.exp()
        encoded = torch.cat([
            t,
            torch.sin(2 * torch.pi * doy / T_year),
            torch.cos(2 * torch.pi * doy / T_year),
            torch.sin(2 * torch.pi * dow / T_week),
            torch.cos(2 * torch.pi * dow / T_week),
        ], dim=-1)
        return self.proj(encoded)


class FourierEncoder(nn.Module):
    """
    Learnable multi-frequency Fourier temporal encoder.

    A learnable frequency matrix W in R^{Kx3} maps the 3-dim raw time
    vector to K frequency projections. sin and cos of each projection
    produce 2K Fourier features capable of capturing arbitrary periodicities
    beyond fixed annual/weekly cycles.

    The original 3 raw features are preserved via concatenation:
      input_to_mlp = [time_features (3), sin(W*t) (K), cos(W*t) (K)]
    total input dim = 3 + 2K.
    """

    def __init__(self, out_dim: int = 16, hidden_dim: int = 32, num_frequencies: int = 8):
        """
        Parameters
        ----------
        out_dim : int
            Output embedding dimension.
        hidden_dim : int
            Hidden width of the MLP following the Fourier feature map.
        num_frequencies : int
            Number of learnable frequency channels K; input to MLP is 3 + 2K.
        """
        super().__init__()
        # Small-magnitude init to keep early-training Fourier features smooth
        self.W = nn.Parameter(torch.randn(num_frequencies, 3) * 0.01)
        input_dim = 3 + 2 * num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable Fourier feature map and project to out_dim.

        Parameters
        ----------
        time_features : torch.Tensor of shape (B, 3)
            Columns: [t_linear, doy, dow].

        Returns
        -------
        torch.Tensor of shape (B, out_dim)
        """
        z = time_features @ self.W.T                              # (B, K)
        fourier = torch.cat([torch.sin(z), torch.cos(z)], dim=-1) # (B, 2K)
        x = torch.cat([time_features, fourier], dim=-1)           # (B, 3+2K)
        return self.proj(x)


class RiemannianProductEncoder(nn.Module):
    """
    Truly manifold-aware product temporal encoder: R x S^1 x S^1 -> S^{out-1}.

    Each circular component (annual, weekly cycle) is embedded on S^1 and
    processed by a SphericalLinear layer - a learnable multi-frequency
    mapping that stays intrinsic to the circle.  The linear time component
    is embedded in Euclidean space.  All components are blended in the
    tangent space of the output hypersphere.

    Concretely, SphericalLinear(2, d) applied to (sin theta, cos theta) in S^1 is
    equivalent to learning a bank of (d-1) independent frequencies for theta,
    then mapping the result onto S^{d-1} via the exp map - a strictly
    more expressive and geometrically principled replacement for the fixed
    sin/cos + MLP pattern in ProductManifoldEncoder.

    Architecture
    ------------
    doy  ->  (sin, cos) in S^1  ->  SphericalLinear(2, hidden_dim)  ->  S^{hidden-1}
    dow  ->  (sin, cos) in S^1  ->  SphericalLinear(2, hidden_dim)  ->  S^{hidden-1}
    t_linear  ->  Linear(1, hidden_dim//4)  ->  R^{t_dim}

    log_north(z_ann)[..., :-1]   in R^{hidden-1}   (tangent coords)
    log_north(z_wk) [..., :-1]   in R^{hidden-1}
    z_t                          in R^{t_dim}

    -> Linear(2*(hidden-1) + t_dim, out-1)
    -> exp_north  ->  S^{out-1}
    """

    def __init__(self, out_dim: int = 16, hidden_dim: int = 32):
        """
        Parameters
        ----------
        out_dim : int
            Output embedding dimension; output lives on S^{out_dim-1}.
        hidden_dim : int
            Intermediate sphere dimension for the annual and weekly SphericalLinear layers.
        """
        super().__init__()
        t_dim = max(hidden_dim // 4, 1)
        self.annual_layer = SphericalLinear(2, hidden_dim)      # S^1 -> S^{hidden-1}
        self.weekly_layer = SphericalLinear(2, hidden_dim)      # S^1 -> S^{hidden-1}
        self.t_embed      = nn.Linear(1, t_dim)                 # t  -> R^{t_dim}
        blend_in = 2 * (hidden_dim - 1) + t_dim
        self.blend = nn.Linear(blend_in, out_dim - 1)

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Encode time features on a product manifold R x S^{hidden-1} x S^{hidden-1}.

        Parameters
        ----------
        time_features : torch.Tensor of shape (B, 3)
            Columns: [t_linear, doy, dow].

        Returns
        -------
        torch.Tensor of shape (B, out_dim) on S^{out_dim-1}
        """
        t   = time_features[:, 0:1]   # (B, 1)
        doy = time_features[:, 1:2]   # (B, 1)
        dow = time_features[:, 2:3]   # (B, 1)

        # Embed circular inputs on S^1 subset of  R^2
        s1_ann = torch.cat([
            torch.sin(2 * torch.pi * doy / _PERIOD_ANNUAL),
            torch.cos(2 * torch.pi * doy / _PERIOD_ANNUAL),
        ], dim=-1)                                              # (B, 2)
        s1_wk = torch.cat([
            torch.sin(2 * torch.pi * dow / _PERIOD_WEEKLY),
            torch.cos(2 * torch.pi * dow / _PERIOD_WEEKLY),
        ], dim=-1)                                              # (B, 2)

        # SphericalLinear: each S^1 -> S^{hidden-1}
        z_ann = self.annual_layer(s1_ann)                       # (B, hidden_dim)
        z_wk  = self.weekly_layer(s1_wk)                       # (B, hidden_dim)

        # Project to tangent space for blending
        v_ann = log_north(z_ann)[..., :-1]                     # (B, hidden-1)
        v_wk  = log_north(z_wk)[..., :-1]                      # (B, hidden-1)
        z_t   = self.t_embed(t)                                # (B, t_dim)

        combined = torch.cat([v_ann, v_wk, z_t], dim=-1)       # (B, blend_in)
        u_tan    = self.blend(combined)                         # (B, out-1)
        u        = torch.cat([u_tan,
                              torch.zeros_like(u_tan[..., :1])], dim=-1)
        return exp_north(u)                                     # (B, out_dim) on S^{out-1}


# =========================================================================
# Factory
# =========================================================================

_TEMPORAL_REGISTRY: dict[str, type] = {
    "product_manifold":   ProductManifoldEncoder,
    "learnable_period":   LearnablePeriodEncoder,
    "fourier":            FourierEncoder,
    "riemannian_product": RiemannianProductEncoder,
}


def build_temporal_encoder(cfg: dict) -> nn.Module:
    """
    Instantiate a temporal encoder from a config dict.

    Parameters
    ----------
    cfg : temporal sub-config (model.temporal in YAML)
    """
    encoder_type = cfg.get("type", "product_manifold")
    if encoder_type not in _TEMPORAL_REGISTRY:
        raise ValueError(
            f"Unknown temporal encoder type '{encoder_type}'. "
            f"Available: {sorted(_TEMPORAL_REGISTRY)}"
        )
    kwargs = dict(
        out_dim=cfg.get("out_dim", 16),
        hidden_dim=cfg.get("hidden_dim", 32),
    )
    if encoder_type == "fourier":
        kwargs["num_frequencies"] = cfg.get("num_frequencies", 8)
    return _TEMPORAL_REGISTRY[encoder_type](**kwargs)
