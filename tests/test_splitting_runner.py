"""Tests for ensure_splits's idempotency (skip/force) logic and strategy
dispatch (random vs. temporal). Splitter itself is stubbed out here -- this
file tests the runner's decisions, not splitting logic (already covered by
test_splitter.py).
"""

import pytest

import src.runners.splitting_runner as runner_mod
from src.runners.splitting_runner import ensure_splits


class _FakeSplitter:
    instances = []

    def __init__(self, sample_name):
        self.sample_name = sample_name
        _FakeSplitter.instances.append(self)
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Redirect SPLITS_DATA to tmp_path and stub out Splitter."""
    splits_dir = tmp_path / "splits"
    monkeypatch.setattr(runner_mod, "SPLITS_DATA", splits_dir)
    _FakeSplitter.instances = []
    monkeypatch.setattr(runner_mod, "Splitter", _FakeSplitter)
    return splits_dir


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


class TestEnsureSplits:
    def test_skips_when_train_test_exist_and_valid_not_requested(self, patched):
        """valid_size=None must not require a valid file for the skip check."""
        splits_dir = patched
        out_dir = splits_dir / "sample"
        _touch(out_dir / "train_default.parquet")
        _touch(out_dir / "test_default.parquet")

        result = ensure_splits(
            sample_name="sample", cleaned_filename="processed_sample",
            valid_size=None, random_state=42,
        )

        assert result == {"skipped": True, "sample_name": "sample", "tag": "default"}
        assert _FakeSplitter.instances == []

    def test_requires_valid_file_when_valid_size_set(self, patched):
        """valid_size not None must require a valid file to be present for the skip to fire."""
        splits_dir = patched
        out_dir = splits_dir / "sample"
        _touch(out_dir / "train_default.parquet")
        _touch(out_dir / "test_default.parquet")
        # no valid_default.parquet

        result = ensure_splits(
            sample_name="sample", cleaned_filename="processed_sample",
            valid_size=0.1, random_state=42,
        )

        assert result["skipped"] is False
        assert len(_FakeSplitter.instances) == 1

    def test_force_reruns_even_if_complete(self, patched):
        """force=True must rebuild splits regardless of existing files."""
        splits_dir = patched
        out_dir = splits_dir / "sample"
        _touch(out_dir / "train_default.parquet")
        _touch(out_dir / "test_default.parquet")

        result = ensure_splits(
            sample_name="sample", cleaned_filename="processed_sample",
            valid_size=None, random_state=42, force=True,
        )

        assert result["skipped"] is False

    def test_sort_by_time_selects_temporal_strategy(self, patched):
        """Passing sort_by_time must dispatch Splitter.run with strategy='temporal'."""
        ensure_splits(
            sample_name="sample", cleaned_filename="processed_sample",
            valid_size=None, random_state=42, sort_by_time="Day",
        )
        kwargs = _FakeSplitter.instances[0].run_kwargs
        assert kwargs["strategy"] == "temporal"
        assert kwargs["time_column"] == "Day"

    def test_no_sort_by_time_selects_random_strategy(self, patched):
        """Omitting sort_by_time must dispatch Splitter.run with strategy='random'."""
        ensure_splits(
            sample_name="sample", cleaned_filename="processed_sample",
            valid_size=None, random_state=42,
        )
        kwargs = _FakeSplitter.instances[0].run_kwargs
        assert kwargs["strategy"] == "random"
        assert kwargs["time_column"] is None

    def test_tag_used_to_namespace_expected_files(self, patched):
        """A custom tag must be reflected in the expected output filenames checked for skipping."""
        splits_dir = patched
        out_dir = splits_dir / "sample"
        _touch(out_dir / "train_custom_tag.parquet")
        _touch(out_dir / "test_custom_tag.parquet")

        result = ensure_splits(
            sample_name="sample", cleaned_filename="processed_sample",
            tag="custom_tag", valid_size=None, random_state=42,
        )
        assert result == {"skipped": True, "sample_name": "sample", "tag": "custom_tag"}
