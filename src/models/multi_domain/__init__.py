# src/models/multi_domain/__init__.py
"""
Multi-domain geometric event classifier.

This subpackage implements the composite representation framework
phi(e) = Psi(phi_who, phi_when, phi_where), where each event domain
(actors, geography, time) is encoded by a configurable Riemannian or
Euclidean encoder and combined via a configurable fusion mechanism:

  actor_encoders.py    - WHO/WHOM: GNN and attribute-only actor encoders
                          (build_actor_encoder).
  geo_encoders.py       - WHERE: hyperspherical and Euclidean geo encoders
                          (build_geo_encoder).
  temporal_encoders.py  - WHEN: product-manifold, Fourier, and Riemannian
                          temporal encoders (build_temporal_encoder).
  fusion.py              - Psi: concat/attention/gated/geometry-aware fusion
                          mechanisms (build_fusion).
  riemannian.py          - Shared Riemannian primitives (log/exp maps,
                          SphericalLinear, SphereReLU) used across encoders.
  model.py                - MultiDomainGeometricModel, the top-level module
                          wiring the four factories together.
  dataset.py / datamodule.py - Data loading utilities feeding all three
                          domains plus the classification target.

All architectural choices are driven by a YAML config dict, enabling
systematic comparison of geometric inductive biases without code changes.
"""

from .actor_encoders import build_actor_encoder
from .fusion import build_fusion
from .geo_encoders import build_geo_encoder
from .model import MultiDomainGeometricModel
from .temporal_encoders import build_temporal_encoder

__all__ = [
    "MultiDomainGeometricModel",
    "build_actor_encoder",
    "build_geo_encoder",
    "build_temporal_encoder",
    "build_fusion",
]
