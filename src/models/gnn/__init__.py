# src/models/gnn/__init__.py
"""
Graph neural network baselines for CAMEO QuadClass prediction.

This subpackage provides two GNN families operating on event/actor graphs:

  homogeneous.py   - HomogeneousGNN, supporting SAGE, GIN, and GAT
                      convolutions on a single-typed graph.
  heterogeneous.py - HeterogeneousGNN, supporting RGCN, RGAT, and HAN
                      convolutions on typed (multi-relation) graphs built
                      from HeteroData.

Both classes share the ``TabularInputEncoder`` (see
``src/models/tabular_encoder.py``) to turn categorical/numeric event
attributes into node features when the graph is not featureless.
"""
