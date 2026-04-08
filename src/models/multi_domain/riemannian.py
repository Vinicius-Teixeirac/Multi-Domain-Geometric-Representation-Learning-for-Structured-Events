# src/models/multi_domain/riemannian.py
"""
Shared Riemannian building blocks for hyperspherical encoders.

All operations use the north pole  p = (0, ..., 0, 1)  as the reference
point.  This choice makes the log and exp maps have closed-form expressions
that are cheap to compute and numerically stable.

Log map at the north pole
-------------------------
For x ∈ S^{d-1} with last coordinate cos θ = x_{-1}:

    log_p(x) = θ/sin(θ) · [x_{:-1}, 0]

Since the last component of any tangent vector at p is 0 by definition
(T_p S^{d-1} = {v : v_d = 0}), it is stored explicitly as 0 and dropped
when operating in tangent-space coordinates.

Exp map at the north pole
-------------------------
For v ∈ T_p S^{d-1} with v_{-1} = 0:

    exp_p(v) = [sin(||v||)/||v|| · v_{:-1}, cos(||v||)]

SphericalLinear
---------------
Maps S^{in-1} → S^{out-1} intrinsically:
  1. log_p(x)    → tangent coords  v ∈ ℝ^{in-1}   (last coord dropped)
  2. W·v + b     → tangent coords  u ∈ ℝ^{out-1}   W ∈ ℝ^{(out-1)×(in-1)}
  3. exp_q(u)    → output point    y ∈ S^{out-1}   (zero re-appended)

SphereReLU
----------
ReLU applied in the tangent space at p.  Since v_{-1} = 0 and ReLU(0) = 0,
the tangent constraint is preserved exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Riemannian primitives
# =========================================================================

def log_north(x: torch.Tensor) -> torch.Tensor:
    """
    Riemannian log map at the north pole p = (0,...,0,1).

    Returns a tangent vector whose last coordinate is identically 0.

    Args:
        x : (*, d) points on S^{d-1}
    Returns:
        (*, d) tangent vectors at p  [last coord = 0]
    """
    inner  = x[..., -1:].clamp(-1 + 1e-7, 1 - 1e-7)           # (*, 1) = cos θ
    u_body = x[..., :-1]                                         # (*, d-1) = sin θ · direction
    u_norm = u_body.norm(dim=-1, keepdim=True).clamp_min(1e-7)  # (*, 1) = |sin θ|
    angle  = torch.acos(inner)                                   # (*, 1) = θ
    return torch.cat([
        angle / u_norm * u_body,
        torch.zeros_like(x[..., -1:]),
    ], dim=-1)                                                    # (*, d)


def exp_north(v: torch.Tensor) -> torch.Tensor:
    """
    Riemannian exp map at the north pole p = (0,...,0,1).

    Expects v[..., -1] = 0 (tangent constraint at north pole).

    Args:
        v : (*, d) tangent vectors at p
    Returns:
        (*, d) points on S^{d-1}
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-7)       # (*, 1)
    body   = torch.sin(v_norm) / v_norm * v[..., :-1]           # (*, d-1)
    last   = torch.cos(v_norm)                                   # (*, 1)
    return torch.cat([body, last], dim=-1)                       # (*, d)


# =========================================================================
# Spherical building blocks
# =========================================================================

class SphericalLinear(nn.Module):
    """
    Linear layer that maps S^{in_dim-1} → S^{out_dim-1} intrinsically.

    Every input and output lies on a hypersphere.

    Algorithm
    ---------
    1. log at source north pole → tangent coords v ∈ ℝ^{in-1}
    2. W·v + b  (W ∈ ℝ^{(out-1)×(in-1)}, b ∈ ℝ^{out-1})
    3. exp at target north pole → point on S^{out-1}

    Initialisation: semi-orthogonal W (preserves tangent-vector norms
    on average, stabilising early-training gradient flow).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        tan_in  = in_dim  - 1
        tan_out = out_dim - 1
        self.weight = nn.Parameter(torch.empty(tan_out, tan_in))
        self.bias   = nn.Parameter(torch.zeros(tan_out))
        if min(tan_in, tan_out) >= 1:
            nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_tan = log_north(x)[..., :-1]                # (B, in-1)
        u_tan = v_tan @ self.weight.T + self.bias      # (B, out-1)
        u     = torch.cat([u_tan,
                           torch.zeros_like(u_tan[..., :1])], dim=-1)
        return exp_north(u)                            # (B, out_dim)


class SphereReLU(nn.Module):
    """
    ReLU applied in the tangent space at the north pole.

    The tangent constraint (last coord = 0) is preserved because ReLU(0) = 0.
    The output is mapped back to S^{d-1} via the exp map.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_tan = log_north(x)[..., :-1]                 # (B, d-1)
        v_tan = F.relu(v_tan)
        v     = torch.cat([v_tan,
                           torch.zeros_like(v_tan[..., :1])], dim=-1)
        return exp_north(v)                             # (B, d)
