"""Tests for the pure helper functions behind graph-entity-ID construction."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.entity_construction import _safe_str, _safe_float, _concat_non_null
from src.utils.constants import NULL_TOKEN


class TestSafeStr:
    def test_regular_value(self):
        """A normal value must stringify as-is."""
        assert _safe_str("USA") == "USA"

    def test_nan_returns_empty_string(self):
        """NaN must become an empty string, not the literal 'nan'."""
        assert _safe_str(np.nan) == ""

    def test_none_returns_empty_string(self):
        """None must become an empty string."""
        assert _safe_str(None) == ""


class TestSafeFloat:
    def test_rounds_to_given_decimals(self):
        """Floats must be rounded and stringified to the requested precision."""
        assert _safe_float(12.34567, decimals=3) == "12.346"

    def test_nan_returns_empty_string(self):
        """NaN must become an empty string, not 'nan'."""
        assert _safe_float(np.nan) == ""

    def test_default_decimals_is_three(self):
        """The default rounding precision must be 3 decimal places."""
        assert _safe_float(1.0 / 3) == "0.333"


class TestConcatNonNull:
    def test_joins_with_hyphen(self):
        """Non-empty tokens must be joined with a hyphen."""
        assert _concat_non_null(["a", "b", "c"]) == "a-b-c"

    def test_drops_empty_strings(self):
        """Empty-string tokens (from missing values) must be excluded from the join."""
        assert _concat_non_null(["a", "", "c"]) == "a-c"

    def test_drops_null_token(self):
        """The explicit NULL_TOKEN sentinel must also be excluded, not treated as real data."""
        assert _concat_non_null(["a", NULL_TOKEN, "c"]) == "a-c"

    def test_all_missing_gives_empty_string(self):
        """If every token is missing, the composite ID must be an empty string (not '--')."""
        assert _concat_non_null(["", "", NULL_TOKEN]) == ""
