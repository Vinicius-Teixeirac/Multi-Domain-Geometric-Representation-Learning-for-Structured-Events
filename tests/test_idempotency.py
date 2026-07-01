"""Tests for the idempotency checks runners use to skip already-completed experiments."""

import json

import pytest

import src.utils.idempotency as idem


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    """Redirect ARTIFACTS_DATA/RESULTS_DIR to a temp directory for isolated tests."""
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"
    artifacts_dir.mkdir()
    results_dir.mkdir()
    monkeypatch.setattr(idem, "ARTIFACTS_DATA", artifacts_dir)
    monkeypatch.setattr(idem, "RESULTS_DIR", results_dir)
    return artifacts_dir, results_dir


class TestFindCheckpoint:
    def test_finds_checkpoint_under_any_model_subdir(self, patched_dirs):
        """_find_checkpoint must search across model-family subdirectories, not just one."""
        artifacts_dir, _ = patched_dirs
        ck_dir = artifacts_dir / "my_dataset" / "models" / "EventMLP" / "exp123"
        ck_dir.mkdir(parents=True)
        (ck_dir / "best_model.pt").write_bytes(b"fake")

        found = idem._find_checkpoint("my_dataset", "exp123")
        assert found == ck_dir / "best_model.pt"

    def test_returns_none_when_missing(self, patched_dirs):
        """No matching checkpoint anywhere must return None, not raise."""
        assert idem._find_checkpoint("my_dataset", "exp123") is None

    def test_returns_none_for_empty_exp_id(self, patched_dirs):
        """An empty exp_id must short-circuit to None (never treated as a wildcard match)."""
        assert idem._find_checkpoint("my_dataset", "") is None


class TestFindResultsJson:
    def test_matches_exact_exp_id_field(self, patched_dirs):
        """_find_results_json must match exp_id as an exact JSON field value."""
        _, results_dir = patched_dirs
        out_dir = results_dir / "my_dataset" / "EventMLP"
        out_dir.mkdir(parents=True)
        (out_dir / "mlp_results_1.json").write_text(json.dumps({"exp_id": "exp123", "x": 1}))

        found = idem._find_results_json("my_dataset", "exp123")
        assert found == out_dir / "mlp_results_1.json"

    def test_does_not_match_substring(self, patched_dirs):
        """A results file whose exp_id is only a substring match must NOT be treated as a hit."""
        _, results_dir = patched_dirs
        out_dir = results_dir / "my_dataset" / "EventMLP"
        out_dir.mkdir(parents=True)
        (out_dir / "mlp_results_1.json").write_text(json.dumps({"exp_id": "exp123456"}))

        assert idem._find_results_json("my_dataset", "exp123") is None

    def test_skips_malformed_json_without_crashing(self, patched_dirs):
        """A corrupt JSON file in the results tree must be skipped, not raise."""
        _, results_dir = patched_dirs
        out_dir = results_dir / "my_dataset" / "EventMLP"
        out_dir.mkdir(parents=True)
        (out_dir / "broken.json").write_text("{not valid json")
        (out_dir / "good.json").write_text(json.dumps({"exp_id": "exp123"}))

        found = idem._find_results_json("my_dataset", "exp123")
        assert found == out_dir / "good.json"


class TestShouldSkip:
    def test_does_not_skip_on_checkpoint_alone(self, patched_dirs):
        """A checkpoint with no matching results JSON must NOT be skipped.

        Regression test: train_model() checkpoints after the first epoch
        that improves, so a run that then crashes before evaluation/results
        are written (e.g. a CUDA OOM while reloading the checkpoint for
        eval) must be retried, not silently treated as already done.
        """
        artifacts_dir, _ = patched_dirs
        ck_dir = artifacts_dir / "my_dataset" / "models" / "EventMLP" / "exp123"
        ck_dir.mkdir(parents=True)
        (ck_dir / "best_model.pt").write_bytes(b"fake")

        skip, info = idem.should_skip("exp123", "my_dataset")
        assert skip is False
        assert info["results_file"] is None

    def test_skips_when_only_results_exist(self, patched_dirs):
        """A found results JSON (no checkpoint) should also report skip=True."""
        _, results_dir = patched_dirs
        out_dir = results_dir / "my_dataset" / "EventMLP"
        out_dir.mkdir(parents=True)
        (out_dir / "mlp_results_1.json").write_text(json.dumps({"exp_id": "exp123"}))

        skip, info = idem.should_skip("exp123", "my_dataset")
        assert skip is True
        assert info["results_file"] is not None

    def test_skips_and_reports_checkpoint_when_both_exist(self, patched_dirs):
        """A completed run (checkpoint + matching results) should skip with both paths populated."""
        artifacts_dir, results_dir = patched_dirs
        ck_dir = artifacts_dir / "my_dataset" / "models" / "EventMLP" / "exp123"
        ck_dir.mkdir(parents=True)
        (ck_dir / "best_model.pt").write_bytes(b"fake")
        out_dir = results_dir / "my_dataset" / "EventMLP"
        out_dir.mkdir(parents=True)
        (out_dir / "mlp_results_1.json").write_text(json.dumps({"exp_id": "exp123"}))

        skip, info = idem.should_skip("exp123", "my_dataset")
        assert skip is True
        assert info["checkpoint"] is not None
        assert info["results_file"] is not None

    def test_does_not_skip_when_nothing_found(self, patched_dirs):
        """No artifacts anywhere must report skip=False."""
        skip, info = idem.should_skip("exp123", "my_dataset")
        assert skip is False
        assert info == {"checkpoint": None, "results_file": None}

    def test_empty_exp_id_never_skips(self, patched_dirs):
        """An empty exp_id must never trigger a skip, even if matching artifacts happen to exist."""
        skip, info = idem.should_skip("", "my_dataset")
        assert skip is False
