# src/representation/text/text_pipeline.py

import torch
import pandas as pd
from transformers import AutoTokenizer
from typing import Dict, Tuple


class TextPipeline:
    """
    Tokenization pipeline for BERT-style models.

    Input:
        DataFrame with columns:
            - text: str
            - label: int

    Output:
        encodings: Dict[str, Tensor]
        labels: Tensor
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier for the AutoTokenizer.
        max_length : int
            Maximum token sequence length; longer inputs are truncated.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Convert a dataframe with 'text' and 'label' columns
        into BERT encodings and label tensor.
        """

        self._validate_df(df)

        texts = df["text"].tolist()
        labels = torch.tensor(df["label"].values, dtype=torch.long)

        encodings = self._tokenize(texts)

        return encodings, labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        """Tokenise a list of strings and return a plain dict of tensors."""
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # IMPORTANT: convert BatchEncoding -> plain dict
        return {k: v for k, v in enc.items()}

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        """Raise ValueError if df is missing the required 'text' or 'label' columns."""
        required = {"text", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for text pipeline: {missing}"
            )
