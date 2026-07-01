# src/models/bert/__init__.py
"""
BERT-based text classifier baseline for CAMEO QuadClass prediction.

This subpackage fine-tunes a pretrained HuggingFace transformer
(``BertForQuadClass`` in model.py) on event text fields, with supporting
``BertDataset`` and datamodule utilities for tokenized encodings, labels,
and DataLoader construction. It serves as a text-only baseline against
which the tabular and geometric multi-domain models are compared.
"""
