"""Tests for reading embedding vocab sizes from fitted encoder artifacts."""

import pytest

import src.utils.categorical_cardinalities as cc
from src.representation.tabular.encoding import SafeLabelEncoder, HashEncoder


@pytest.fixture
def fake_schema(monkeypatch):
    """Replace ENCODING_SCHEMA with a small fixture-controlled schema."""
    schema = {
        "label_col": {"type": "categorical", "method": "label"},
        "hash_col": {"type": "categorical", "method": "hash", "params": {"hash_dim": 1024}},
    }
    monkeypatch.setattr(cc, "ENCODING_SCHEMA", schema)
    return schema


class TestLoadCategoricalCardinalities:
    def test_reads_label_and_hash_cardinalities(self, tmp_path, fake_schema):
        """Label encoders report num_classes_; hash encoders report num_buckets."""
        import pandas as pd

        label_enc = SafeLabelEncoder().fit(pd.Series(["a", "b", "c"]))
        label_enc.save(tmp_path / "label_col.json")

        hash_enc = HashEncoder(num_buckets=1024)
        hash_enc.save(tmp_path / "hash_col.json")

        result = cc.load_categorical_cardinalities(["label_col", "hash_col"], tmp_path)
        assert result["label_col"] == 3
        assert result["hash_col"] == 1024

    def test_skips_column_with_no_schema_entry(self, tmp_path, fake_schema):
        """A column absent from ENCODING_SCHEMA must be silently skipped, not raise."""
        result = cc.load_categorical_cardinalities(["unknown_col"], tmp_path)
        assert result == {}

    def test_skips_column_with_missing_artifact_file(self, tmp_path, fake_schema):
        """A schema entry with no artifact file on disk yet must be silently skipped."""
        result = cc.load_categorical_cardinalities(["label_col"], tmp_path)
        assert result == {}

    def test_partial_results_when_some_artifacts_missing(self, tmp_path, fake_schema):
        """Columns with artifacts present must still be returned even if others are missing."""
        hash_enc = HashEncoder(num_buckets=1024)
        hash_enc.save(tmp_path / "hash_col.json")

        result = cc.load_categorical_cardinalities(["label_col", "hash_col"], tmp_path)
        assert result == {"hash_col": 1024}
