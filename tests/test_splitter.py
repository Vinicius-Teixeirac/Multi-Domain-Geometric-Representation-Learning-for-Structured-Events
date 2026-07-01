"""Tests for Splitter: random/temporal strategies, validation-split skip
logic, and metadata persistence."""

import json

import pandas as pd
import pytest

import src.preprocessing.splitting as splitting_mod
from src.preprocessing.splitting import Splitter


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    """Redirect PROCESSED_DATA/SPLITS_DATA to a temp directory for isolated tests."""
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    monkeypatch.setattr(splitting_mod, "PROCESSED_DATA", processed_dir)
    monkeypatch.setattr(splitting_mod, "SPLITS_DATA", splits_dir)
    return processed_dir, splits_dir


def _write_clean_df(processed_dir, sample_name: str, n: int) -> None:
    dataset_dir = processed_dir / sample_name
    dataset_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "id": range(n),
        "QuadClass": [i % 4 for i in range(n)],
        "Day": [20200101 + i for i in range(n)],
    })
    df.to_parquet(dataset_dir / "processed_sample.parquet", index=False)


class TestSortByTime:
    def test_sorts_ascending(self):
        """sort_by_time must return rows in ascending order of the given column."""
        df = pd.DataFrame({"Day": [3, 1, 2]})
        out = Splitter.sort_by_time(df, "Day")
        assert out["Day"].tolist() == [1, 2, 3]

    def test_missing_column_raises(self):
        """A nonexistent time column must raise ValueError, not a KeyError deep inside pandas."""
        df = pd.DataFrame({"Day": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            Splitter.sort_by_time(df, "not_a_column")


class TestSplitterRun:
    def test_random_split_produces_train_valid_test(self, patched_dirs):
        """A large-enough dataset with valid_size set must produce all three split files."""
        processed_dir, splits_dir = patched_dirs
        _write_clean_df(processed_dir, "sample", n=3000)

        splitter = Splitter(sample_name="sample")
        splitter.run(
            filename="processed_sample", strategy="random",
            test_size=0.15, valid_size=0.10, stratify_by="QuadClass",
            random_state=42, tag="default",
        )

        out_dir = splits_dir / "sample"
        assert (out_dir / "train_default.parquet").exists()
        assert (out_dir / "valid_default.parquet").exists()
        assert (out_dir / "test_default.parquet").exists()

        train = pd.read_parquet(out_dir / "train_default.parquet")
        valid = pd.read_parquet(out_dir / "valid_default.parquet")
        test = pd.read_parquet(out_dir / "test_default.parquet")
        assert len(train) + len(valid) + len(test) == 3000
        # no event overlaps across splits (event-level inductive split)
        assert set(train["id"]) & set(valid["id"]) & set(test["id"]) == set()

    def test_valid_size_none_skips_validation_split(self, patched_dirs):
        """valid_size=None must produce only train/test, with metadata recording valid_size=None."""
        processed_dir, splits_dir = patched_dirs
        _write_clean_df(processed_dir, "sample", n=100)

        splitter = Splitter(sample_name="sample")
        splitter.run(
            filename="processed_sample", strategy="random",
            test_size=0.2, valid_size=None, random_state=42, tag="default",
        )

        out_dir = splits_dir / "sample"
        assert (out_dir / "train_default.parquet").exists()
        assert not (out_dir / "valid_default.parquet").exists()

        meta = json.loads((out_dir / "split_default.json").read_text())
        assert meta["valid_size"] is None

    def test_small_dataset_skips_validation_split_with_warning(self, patched_dirs):
        """Below _MIN_TRAIN_FOR_VALID, even a requested valid_size must be skipped with a warning."""
        processed_dir, splits_dir = patched_dirs
        _write_clean_df(processed_dir, "sample", n=100)

        splitter = Splitter(sample_name="sample")
        with pytest.warns(UserWarning, match="skipping validation split"):
            splitter.run(
                filename="processed_sample", strategy="random",
                test_size=0.15, valid_size=0.10, random_state=42, tag="default",
            )

        out_dir = splits_dir / "sample"
        assert not (out_dir / "valid_default.parquet").exists()

    def test_temporal_strategy_requires_time_column(self, patched_dirs):
        """strategy='temporal' without time_column must raise ValueError."""
        processed_dir, splits_dir = patched_dirs
        _write_clean_df(processed_dir, "sample", n=100)
        splitter = Splitter(sample_name="sample")
        with pytest.raises(ValueError, match="requires time_column"):
            splitter.run(filename="processed_sample", strategy="temporal", random_state=42)

    def test_random_strategy_rejects_time_column(self, patched_dirs):
        """strategy='random' combined with a time_column must raise ValueError."""
        processed_dir, splits_dir = patched_dirs
        _write_clean_df(processed_dir, "sample", n=100)
        splitter = Splitter(sample_name="sample")
        with pytest.raises(ValueError, match="not allowed"):
            splitter.run(
                filename="processed_sample", strategy="random",
                time_column="Day", random_state=42,
            )

    def test_temporal_strategy_ignores_stratify_with_warning(self, patched_dirs):
        """Temporal splitting must ignore stratify_by (shuffling would break chronological order)."""
        processed_dir, splits_dir = patched_dirs
        _write_clean_df(processed_dir, "sample", n=3000)
        splitter = Splitter(sample_name="sample")
        with pytest.warns(UserWarning, match="Stratification ignored"):
            splitter.run(
                filename="processed_sample", strategy="temporal",
                time_column="Day", stratify_by="QuadClass", valid_size=None,
                random_state=42,
            )

    def test_unknown_strategy_raises(self, patched_dirs):
        """An unrecognized strategy string must raise ValueError."""
        processed_dir, splits_dir = patched_dirs
        _write_clean_df(processed_dir, "sample", n=100)
        splitter = Splitter(sample_name="sample")
        with pytest.raises(ValueError, match="Unknown split strategy"):
            splitter.run(filename="processed_sample", strategy="bogus", random_state=42)

    def test_output_dir_created_on_construction(self, patched_dirs):
        """Splitter.__init__ must create the output directory even before run() is called."""
        _, splits_dir = patched_dirs
        Splitter(sample_name="brand_new_sample")
        assert (splits_dir / "brand_new_sample").exists()
