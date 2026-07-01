"""Tests for ensure_entities's idempotency (skip/force) logic.

build_event_entities itself is stubbed out here -- this file tests the
runner's skip/force decision, not entity construction (already covered by
test_entity_construction.py).
"""

import pytest

import src.runners.entity_runner as runner_mod
from src.runners.entity_runner import ensure_entities


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Redirect ENTITIES_DATA to tmp_path and stub out build_event_entities."""
    entities_dir = tmp_path / "entities"
    monkeypatch.setattr(runner_mod, "ENTITIES_DATA", entities_dir)
    calls = []
    monkeypatch.setattr(
        runner_mod, "build_event_entities",
        lambda dataset_name, split_tag: calls.append((dataset_name, split_tag)),
    )
    return entities_dir, calls


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


class TestEnsureEntities:
    def test_skips_when_train_and_test_exist_no_valid(self, patched):
        """train+test present (no valid file at all) must be treated as complete -> skip."""
        entities_dir, calls = patched
        out_dir = entities_dir / "my_dataset"
        _touch(out_dir / "train_default_entities.parquet")
        _touch(out_dir / "test_default_entities.parquet")

        result = ensure_entities("my_dataset", split_tag="default")

        assert result == {"skipped": True, "dataset": "my_dataset", "split_tag": "default"}
        assert calls == []

    def test_skips_when_valid_also_present(self, patched):
        """If a valid file exists, it becomes required too, but skip should still succeed if present."""
        entities_dir, calls = patched
        out_dir = entities_dir / "my_dataset"
        for name in ["train_default_entities.parquet", "test_default_entities.parquet", "valid_default_entities.parquet"]:
            _touch(out_dir / name)

        result = ensure_entities("my_dataset", split_tag="default")
        assert result["skipped"] is True
        assert calls == []

    def test_runs_when_train_missing(self, patched):
        """A missing required file (train) must trigger a rebuild."""
        entities_dir, calls = patched
        out_dir = entities_dir / "my_dataset"
        _touch(out_dir / "test_default_entities.parquet")

        result = ensure_entities("my_dataset", split_tag="default")

        assert result["skipped"] is False
        assert calls == [("my_dataset", "default")]

    def test_force_rebuilds_even_if_complete(self, patched):
        """force=True must rebuild even when every expected file already exists."""
        entities_dir, calls = patched
        out_dir = entities_dir / "my_dataset"
        _touch(out_dir / "train_default_entities.parquet")
        _touch(out_dir / "test_default_entities.parquet")

        result = ensure_entities("my_dataset", split_tag="default", force=True)

        assert result["skipped"] is False
        assert len(calls) == 1
