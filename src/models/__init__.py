# src/models/__init__.py
"""
Model implementations for GDELT structured-event classification.

This package collects every model family evaluated in the BRACIS 2026 study,
ranging from purely tabular baselines to the geometric multi-domain
architecture that is the paper's main contribution:

  tabular_encoder.py  - TabularInputEncoder, shared categorical/numeric
                         feature encoder reused by mlp/model.py and
                         gnn/heterogeneous.py.
  bert/                - Fine-tuned BERT text classifier baseline
                         (BertForQuadClass).
  gnn/                 - Homogeneous and heterogeneous graph neural network
                         baselines (SAGE/GIN/GAT and RGCN/RGAT/HAN).
  mlp/                 - Plain MLP baseline over tabular event features
                         (EventMLP).
  multi_domain/        - The geometric multi-domain model: per-domain
                         Riemannian/Euclidean encoders for actors, geography,
                         and time, combined via a configurable fusion
                         mechanism (MultiDomainGeometricModel).

Submodules are imported explicitly by callers (e.g.
``from src.models.mlp.model import EventMLP``); this package init does not
eagerly import them to avoid pulling in optional heavy dependencies
(transformers, torch_geometric) unless the corresponding model is used.
"""
