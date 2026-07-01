"""Tests for MissingValueHandler's per-column missing-value policies."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaning import MissingValueHandler
from src.utils.constants import NULL_TOKEN


class TestMissingValueHandler:
    def test_explicit_policy_fills_with_null_token(self):
        """'explicit' policy replaces NaN with NULL_TOKEN and casts the column to string."""
        df = pd.DataFrame({"col": ["a", None, "b"]})
        handler = MissingValueHandler({"col": {"missing": "explicit"}})
        out = handler.apply(df)
        assert out["col"].tolist() == ["a", NULL_TOKEN, "b"]
        assert out["col"].dtype == "string"

    def test_indicator_policy_adds_flag_column_and_keeps_nan(self):
        """'indicator' policy adds a {col}__is_missing flag but leaves the original NaNs in place."""
        df = pd.DataFrame({"col": [1.0, np.nan, 3.0]})
        handler = MissingValueHandler({"col": {"missing": "indicator"}})
        out = handler.apply(df)
        assert out["col__is_missing"].tolist() == [0, 1, 0]
        assert pd.isna(out["col"].iloc[1])

    def test_error_policy_raises_on_missing_values(self):
        """'error' policy must raise ValueError if any value in the column is missing."""
        df = pd.DataFrame({"col": [1, None, 3]})
        handler = MissingValueHandler({"col": {"missing": "error"}})
        with pytest.raises(ValueError, match="col"):
            handler.apply(df)

    def test_error_policy_passes_when_no_missing_values(self):
        """'error' policy must be a no-op when there are no missing values."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        handler = MissingValueHandler({"col": {"missing": "error"}})
        out = handler.apply(df)
        assert out["col"].tolist() == [1, 2, 3]

    def test_column_absent_from_dataframe_is_skipped(self):
        """A schema entry for a column not present in df must be silently ignored."""
        df = pd.DataFrame({"other": [1, 2]})
        handler = MissingValueHandler({"missing_col": {"missing": "error"}})
        out = handler.apply(df)
        pd.testing.assert_frame_equal(out, df)

    def test_original_dataframe_is_not_mutated(self):
        """apply() must return a copy; the caller's original DataFrame stays untouched."""
        df = pd.DataFrame({"col": ["a", None]})
        handler = MissingValueHandler({"col": {"missing": "explicit"}})
        handler.apply(df)
        assert pd.isna(df["col"].iloc[1])

    def test_multiple_columns_with_different_policies(self):
        """Each column's policy must be applied independently within a single call."""
        df = pd.DataFrame({
            "explicit_col": ["a", None],
            "indicator_col": [1.0, np.nan],
        })
        handler = MissingValueHandler({
            "explicit_col": {"missing": "explicit"},
            "indicator_col": {"missing": "indicator"},
        })
        out = handler.apply(df)
        assert out["explicit_col"].iloc[1] == NULL_TOKEN
        assert out["indicator_col__is_missing"].iloc[1] == 1
