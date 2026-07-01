"""Tests for ensure_text's per-split idempotency logic and QuadClass->label
column rename. build_event_texts is stubbed out here for isolation and
speed (already covered by test_text_builder.py)."""

import pandas as pd
import pytest

import src.runners.text_runner as runner_mod
from src.runners.text_runner import ensure_text


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Redirect SPLITS_DATA/TEXT_DATA to tmp_path and stub out build_event_texts."""
    splits_dir = tmp_path / "splits"
    text_dir = tmp_path / "text"
    monkeypatch.setattr(runner_mod, "SPLITS_DATA", splits_dir)
    monkeypatch.setattr(runner_mod, "TEXT_DATA", text_dir)
    calls = []

    def fake_build_event_texts(df, dictionaries):
        calls.append((len(df), dictionaries))
        return [f"sentence {i}" for i in range(len(df))]

    monkeypatch.setattr(runner_mod, "build_event_texts", fake_build_event_texts)
    return splits_dir, text_dir, calls


def _write_split(splits_dir, dataset, split, tag, n):
    dataset_dir = splits_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"QuadClass": [i % 4 for i in range(n)]})
    df.to_parquet(dataset_dir / f"{split}_{tag}.parquet", index=False)


class TestEnsureText:
    def test_builds_text_for_available_splits_only(self, patched):
        """train/test present, valid absent -> only train/test text files are written."""
        splits_dir, text_dir, calls = patched
        _write_split(splits_dir, "my_dataset", "train", "default", 5)
        _write_split(splits_dir, "my_dataset", "test", "default", 3)

        result = ensure_text("my_dataset", "default", dictionaries={})

        out_dir = text_dir / "my_dataset"
        assert (out_dir / "train_default_text.parquet").exists()
        assert (out_dir / "test_default_text.parquet").exists()
        assert not (out_dir / "valid_default_text.parquet").exists()
        assert len(calls) == 2
        assert result == {"skipped": False, "dataset": "my_dataset", "split_tag": "default"}

    def test_quadclass_renamed_to_label(self, patched):
        """Output columns must be exactly ['text', 'label'], with label sourced from QuadClass."""
        splits_dir, text_dir, calls = patched
        _write_split(splits_dir, "my_dataset", "train", "default", 4)

        ensure_text("my_dataset", "default", dictionaries={})

        out = pd.read_parquet(text_dir / "my_dataset" / "train_default_text.parquet")
        assert list(out.columns) == ["text", "label"]
        assert out["label"].tolist() == [0, 1, 2, 3]

    def test_skips_split_with_existing_output(self, patched):
        """A split whose text output already exists must not be rebuilt (per-split skip)."""
        splits_dir, text_dir, calls = patched
        _write_split(splits_dir, "my_dataset", "train", "default", 5)
        out_dir = text_dir / "my_dataset"
        out_dir.mkdir(parents=True)
        pd.DataFrame({"text": ["x"], "label": [0]}).to_parquet(out_dir / "train_default_text.parquet")

        ensure_text("my_dataset", "default", dictionaries={})

        assert calls == []  # build_event_texts never called for the already-built split

    def test_force_rebuilds_existing_split(self, patched):
        """force=True must rebuild a split even if its text output already exists."""
        splits_dir, text_dir, calls = patched
        _write_split(splits_dir, "my_dataset", "train", "default", 5)
        out_dir = text_dir / "my_dataset"
        out_dir.mkdir(parents=True)
        pd.DataFrame({"text": ["x"], "label": [0]}).to_parquet(out_dir / "train_default_text.parquet")

        ensure_text("my_dataset", "default", dictionaries={}, force=True)

        assert len(calls) == 1

    def test_missing_input_split_is_silently_skipped(self, patched):
        """A split with no input parquet at all (e.g. no valid split) must be silently skipped."""
        splits_dir, text_dir, calls = patched
        _write_split(splits_dir, "my_dataset", "train", "default", 5)
        # no valid_default.parquet, no test_default.parquet

        result = ensure_text("my_dataset", "default", dictionaries={})

        assert len(calls) == 1  # only train was buildable
        assert result["skipped"] is False  # always reports not-skipped at the top level

    def test_dictionaries_forwarded_to_build_event_texts(self, patched):
        """The dictionaries argument must be passed through unchanged."""
        splits_dir, text_dir, calls = patched
        _write_split(splits_dir, "my_dataset", "train", "default", 2)
        sentinel = {"EventCode": {"01": "test"}}

        ensure_text("my_dataset", "default", dictionaries=sentinel)

        assert calls[0][1] is sentinel
