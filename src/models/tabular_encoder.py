# src/models/tabular_encoder.py
"""
Shared tabular feature encoder for event attributes.

``TabularInputEncoder`` turns categorical event columns into learned
embeddings and passes numeric columns through unchanged, concatenating the
result into a single dense feature vector. It is not tied to any specific
downstream architecture and is reused by both ``mlp/model.py`` (as the
input stage of ``EventMLP``) and ``gnn/heterogeneous.py`` (to build event
node features from tabular attributes before message passing).
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional

__all__ = ["TabularInputEncoder"]


class TabularInputEncoder(nn.Module):
    """
    Encodes tabular inputs:
        - categorical features via embeddings
        - numeric features via identity

    Input:
        x_cat: Dict[str, LongTensor]  (batch,)
        x_num: FloatTensor            (batch, numeric_dim)

    Output:
        FloatTensor (batch, output_dim)
    """

    def __init__(
        self,
        categorical_cardinalities: Dict[str, int],
        numeric_dim: int,
        embedding_dim_rule: str = "sqrt",
        embedding_dropout: float = 0.2,
    ):
        """
        Parameters
        ----------
        categorical_cardinalities : dict[str, int]
            Mapping from column name to vocabulary size (excluding the unknown
            token, which adds one extra embedding row internally).
        numeric_dim : int
            Number of continuous input features (concatenated after embeddings).
        embedding_dim_rule : str
            Heuristic for computing embedding size from cardinality:
            'sqrt' -> min(128, sqrt(card)); 'log' -> min(128, log2(card)+1).
        embedding_dropout : float
            Dropout probability applied to each categorical embedding.
        """
        super().__init__()

        self.embedding_dropout = embedding_dropout
        self.categorical_cardinalities = categorical_cardinalities
        self.numeric_dim = numeric_dim

        self.embeddings = nn.ModuleDict()
        self.output_dim = numeric_dim

        for name, cardinality in categorical_cardinalities.items():
            emb_dim = self._embedding_dim(cardinality, embedding_dim_rule)
            self.embeddings[name] = nn.Embedding(
                num_embeddings=cardinality + 1,
                embedding_dim=emb_dim,
                padding_idx=0,
            )
            self.output_dim += emb_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_cat: Dict[str, torch.Tensor],
        x_num: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode categorical and numeric inputs into a single dense vector.

        Parameters
        ----------
        x_cat : dict[str, torch.Tensor of shape (B,)]
            Per-column integer indices for categorical features.
        x_num : torch.Tensor of shape (B, numeric_dim) or None
            Continuous features; None or empty tensor is ignored.

        Returns
        -------
        torch.Tensor of shape (B, output_dim)
        """
        parts = []

        for name, emb in self.embeddings.items():
            e = emb(x_cat[name])
            if self.embedding_dropout > 0:
                e = nn.functional.dropout(
                    e, p=self.embedding_dropout, training=self.training
                )
            parts.append(e)

        if x_num is not None and x_num.numel() > 0:
            parts.append(x_num)

        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    @staticmethod
    def _embedding_dim(cardinality: int, rule: str) -> int:
        """Compute embedding size from cardinality using a heuristic rule.

        Both rules are capped at 128 to prevent oversized embeddings for very
        high-cardinality hash columns (e.g. FeatureID with 2^20 buckets).
        """
        if rule == "sqrt":
            return max(1, min(128, int(math.sqrt(cardinality))))
        if rule == "log":
            return max(1, min(128, int(math.log2(cardinality)) + 1))
        raise ValueError(f"Unknown embedding dim rule '{rule}'")
