"""Tests for ensure_cleaned's idempotency (skip/force) logic.

DataCleaner itself is stubbed out here -- this file tests the runner's
skip/force decision, not the cleaning logic (already covered by
test_data_cleaner.py).
"""

from pathlib import Path

import pytest

import src.runners.cleaning_runner as runner_mod
from src.runners.cleaning_runner import ensure_cleaned


class _FakeDataCleaner:
    """Stub replacing DataCleaner: records whether it was constructed/run."""

    instances = []

    def __init__(self, sample_name, columns=None):
        self.sample_name = sample_name
        self.columns = columns
        _FakeDataCleaner.instances.append(self)

    def run(self, sample_name, target_cols=None):
        self.ran_with = (sample_name, target_cols)
        return Path("fake_output.parquet")


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Redirect PROCESSED_DATA to tmp_path and stub out DataCleaner."""
    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(runner_mod, "PROCESSED_DATA", processed_dir)
    _FakeDataCleaner.instances = []
    monkeypatch.setattr(runner_mod, "DataCleaner", _FakeDataCleaner)
    return processed_dir


class TestEnsureCleaned:
    def test_skips_when_output_already_exists(self, patched):
        """An existing cleaned parquet must short-circuit without instantiating DataCleaner."""
        processed_dir = patched
        out_dir = processed_dir / "sample"
        out_dir.mkdir(parents=True)
        expected_path = out_dir / "processed_sample.parquet"
        expected_path.write_bytes(b"fake")

        result = ensure_cleaned(sample_name="sample")

        assert result == expected_path
        assert len(_FakeDataCleaner.instances) == 0

    def test_runs_when_output_missing(self, patched):
        """No existing output must trigger construction and run() of DataCleaner."""
        ensure_cleaned(sample_name="sample", target_cols=["QuadClass"])

        assert len(_FakeDataCleaner.instances) == 1
        assert _FakeDataCleaner.instances[0].ran_with == ("sample", ["QuadClass"])

    def test_force_reruns_even_if_output_exists(self, patched):
        """force=True must re-run cleaning even when the output file already exists."""
        processed_dir = patched
        out_dir = processed_dir / "sample"
        out_dir.mkdir(parents=True)
        (out_dir / "processed_sample.parquet").write_bytes(b"fake")

        ensure_cleaned(sample_name="sample", force=True)

        assert len(_FakeDataCleaner.instances) == 1

    def test_columns_argument_forwarded(self, patched):
        """The `columns` kwarg must be passed through to DataCleaner's constructor."""
        ensure_cleaned(sample_name="sample", columns=["a", "b"])
        assert _FakeDataCleaner.instances[0].columns == ["a", "b"]
