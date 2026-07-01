"""Tests for DataCleaner: column validation, casting, target normalisation,
and the end-to-end run() pipeline. Uses a small fake GDELT-like schema
(monkeypatched into src.preprocessing.cleaning) instead of the real
58-column codebook, so tests stay fast and independent of schema changes.
"""

import numpy as np
import pandas as pd
import pytest

import src.preprocessing.cleaning as cleaning_mod
from src.preprocessing.cleaning import DataCleaner


FAKE_COLUMNS = ["id_col", "int_col", "float_col", "str_col", "target_col"]


@pytest.fixture
def fake_schema(tmp_path, monkeypatch):
    """Redirect DataCleaner's schema/path dependencies to small, controlled values."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    monkeypatch.setattr(cleaning_mod, "RAW_DATA", raw_dir)
    monkeypatch.setattr(cleaning_mod, "PROCESSED_DATA", processed_dir)
    monkeypatch.setattr(cleaning_mod, "GDELT_EVENT_COLUMNS", FAKE_COLUMNS)
    monkeypatch.setattr(cleaning_mod, "CHOSEN_COLUMNS", FAKE_COLUMNS)
    # target_col included so cast_types() promotes it to nullable Int64 before
    # normalize_targets() runs -- mirrors production, where the QuadClass target
    # column is itself in the id/target group cast to Int64 upstream of normalization.
    monkeypatch.setattr(cleaning_mod, "INTEGER_COLUMNS", ["int_col", "target_col"])
    monkeypatch.setattr(cleaning_mod, "FLOAT_COLUMNS", ["float_col"])
    monkeypatch.setattr(cleaning_mod, "STRING_COLUMNS", ["str_col"])
    monkeypatch.setattr(cleaning_mod, "COLUMNS_SCHEMA", {"str_col": {"missing": "explicit"}})
    return raw_dir, processed_dir


class TestInit:
    def test_unknown_column_raises(self, fake_schema):
        """Requesting a column outside GDELT_EVENT_COLUMNS must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown columns"):
            DataCleaner(sample_name="sample", columns=["not_a_real_column"])

    def test_excluded_columns_warn(self, fake_schema):
        """Selecting a strict subset of the full codebook must emit a UserWarning."""
        with pytest.warns(UserWarning):
            DataCleaner(sample_name="sample", columns=["id_col"])

    def test_defaults_to_chosen_columns(self, fake_schema):
        """Omitting 'columns' must default to the full CHOSEN_COLUMNS list."""
        cleaner = DataCleaner(sample_name="sample")
        assert cleaner.selected_columns == FAKE_COLUMNS

    def test_creates_output_directory(self, fake_schema):
        """The processed-data output directory must be created on construction."""
        _, processed_dir = fake_schema
        DataCleaner(sample_name="my_sample")
        assert (processed_dir / "my_sample").exists()


class TestSelectColumns:
    def test_raises_when_input_missing_required_column(self, fake_schema):
        """A required column absent from the input DataFrame must raise ValueError."""
        cleaner = DataCleaner(sample_name="sample")
        df = pd.DataFrame({"id_col": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            cleaner.select_columns(df)

    def test_drops_columns_not_selected(self, fake_schema):
        """Columns outside selected_columns must be dropped from the output."""
        with pytest.warns(UserWarning):
            cleaner = DataCleaner(sample_name="sample", columns=["id_col", "int_col"])
        df = pd.DataFrame({"id_col": [1], "int_col": [2], "extra_col": [3]})
        out = cleaner.select_columns(df)
        assert list(out.columns) == ["id_col", "int_col"]


class TestCastTypes:
    def test_casts_each_group_to_canonical_dtype(self, fake_schema):
        """int_col -> Int64, float_col -> float32, str_col -> string."""
        cleaner = DataCleaner(sample_name="sample")
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1, 2, 3],
            "str_col": [1, 2, 3],
        })
        out = cleaner.cast_types(df)
        assert str(out["int_col"].dtype) == "Int64"
        assert str(out["float_col"].dtype) == "float32"
        assert str(out["str_col"].dtype) == "string"


class TestNormalizeTargets:
    def test_shifts_one_based_to_zero_based(self, fake_schema):
        """Labels starting at 1 must be shifted down by 1 to become zero-based."""
        cleaner = DataCleaner(sample_name="sample")
        df = pd.DataFrame({"target_col": pd.array([1, 2, 3, 4], dtype="Int64")})
        out = cleaner.normalize_targets(df, ["target_col"])
        assert sorted(out["target_col"].unique().tolist()) == [0, 1, 2, 3]

    def test_raises_on_non_integer_dtype(self, fake_schema):
        """A float-typed target column must raise TypeError before any shifting logic runs."""
        cleaner = DataCleaner(sample_name="sample")
        df = pd.DataFrame({"target_col": [1.0, 2.0]})
        with pytest.raises(TypeError):
            cleaner.normalize_targets(df, ["target_col"])

    def test_raises_on_non_contiguous_labels(self, fake_schema):
        """Zero-based labels with a gap (e.g. {0, 2}, missing 1) must raise ValueError."""
        cleaner = DataCleaner(sample_name="sample")
        df = pd.DataFrame({"target_col": pd.array([0, 2, 0, 2], dtype="Int64")})
        with pytest.raises(ValueError, match="not contiguous"):
            cleaner.normalize_targets(df, ["target_col"])

    def test_column_absent_is_skipped(self, fake_schema):
        """A requested target column missing from df must be silently skipped, not raise."""
        cleaner = DataCleaner(sample_name="sample")
        df = pd.DataFrame({"other": [1, 2]})
        out = cleaner.normalize_targets(df, ["target_col"])
        pd.testing.assert_frame_equal(out, df)


class TestSaveAndRun:
    def test_save_writes_processed_prefix(self, fake_schema):
        """save() must write to processed_{filename}.parquet under output_dir."""
        cleaner = DataCleaner(sample_name="sample")
        df = pd.DataFrame({"a": [1]})
        path = cleaner.save(df, "sample")
        assert path.name == "processed_sample.parquet"
        assert path.exists()

    def test_run_end_to_end(self, fake_schema):
        """run() must load raw, select/cast columns, normalise targets, drop missing-target
        rows, apply missing-value policy, and save -- returning the output path."""
        raw_dir, processed_dir = fake_schema
        df = pd.DataFrame({
            "id_col": [1, 2, 3],
            "int_col": [10, 20, 30],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", None, "c"],
            "target_col": [1, 2, None],
        })
        df.to_parquet(raw_dir / "my_sample.parquet", index=False)

        cleaner = DataCleaner(sample_name="my_sample")
        out_path = cleaner.run(sample_name="my_sample", target_cols=["target_col"])

        result = pd.read_parquet(out_path)
        # row with missing target_col must have been dropped
        assert len(result) == 2
        # 1-based -> 0-based shift
        assert sorted(result["target_col"].unique().tolist()) == [0, 1]
        # explicit missing-value policy applied to str_col
        assert (result["str_col"] == "__NULL__").any() or result["str_col"].notna().all()
