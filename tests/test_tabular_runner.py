"""Tests for ensure_tabular_features's idempotency (skip/force) logic.

TabularPipeline itself is stubbed out here -- this file tests the runner's
skip/force decision, not the tabular encoding pipeline.
"""

import pytest

import src.runners.tabular_runner as runner_mod
from src.runners.tabular_runner import ensure_tabular_features


class _FakeTabularPipeline:
    instances = []

    def __init__(self, dataset_name, split_tag):
        self.dataset_name = dataset_name
        self.split_tag = split_tag
        self.ran = False
        _FakeTabularPipeline.instances.append(self)

    def run(self):
        self.ran = True


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Redirect FEATURES_DATA to tmp_path and stub out TabularPipeline."""
    features_dir = tmp_path / "features"
    monkeypatch.setattr(runner_mod, "FEATURES_DATA", features_dir)
    _FakeTabularPipeline.instances = []
    monkeypatch.setattr(runner_mod, "TabularPipeline", _FakeTabularPipeline)
    return features_dir


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


class TestEnsureTabularFeatures:
    def test_skips_when_train_and_test_exist(self, patched):
        """train+test feature files present must short-circuit without running the pipeline."""
        features_dir = patched
        out_dir = features_dir / "my_dataset"
        _touch(out_dir / "train_default_features.parquet")
        _touch(out_dir / "test_default_features.parquet")

        result = ensure_tabular_features("my_dataset")

        assert result == {"skipped": True, "dataset": "my_dataset", "split_tag": "default"}
        assert _FakeTabularPipeline.instances == []

    def test_runs_when_missing(self, patched):
        """Missing feature files must trigger TabularPipeline construction and run()."""
        result = ensure_tabular_features("my_dataset", split_tag="tag1")

        assert result["skipped"] is False
        assert len(_FakeTabularPipeline.instances) == 1
        assert _FakeTabularPipeline.instances[0].ran is True
        assert _FakeTabularPipeline.instances[0].split_tag == "tag1"

    def test_force_reruns_even_if_complete(self, patched):
        """force=True must rerun the pipeline even when both expected files exist."""
        features_dir = patched
        out_dir = features_dir / "my_dataset"
        _touch(out_dir / "train_default_features.parquet")
        _touch(out_dir / "test_default_features.parquet")

        result = ensure_tabular_features("my_dataset", force=True)

        assert result["skipped"] is False
        assert len(_FakeTabularPipeline.instances) == 1
