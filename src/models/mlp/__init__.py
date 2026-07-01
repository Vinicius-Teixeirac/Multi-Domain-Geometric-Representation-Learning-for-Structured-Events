# src/models/mlp/__init__.py
"""
Plain MLP baseline over tabular event features.

This subpackage implements ``EventMLP``, a feed-forward classifier that
consumes categorical (embedded) and numeric event features via the shared
``TabularInputEncoder``, along with the ``EventDataset``/``EventDataModule``
pair that loads TabularPipeline outputs and serves PyTorch DataLoaders.
It serves as the non-relational, non-geometric baseline in the comparison.
"""
