"""Tests for shared file-loading helpers (JSON and parquet)."""

import json

import pandas as pd
import pytest

from src.utils.loading import load_json, load_parquet, load_split


class TestLoadJson:
    def test_round_trip(self, tmp_path):
        """A written JSON file must be read back with identical contents."""
        path = tmp_path / "data.json"
        payload = {"a": 1, "b": [1, 2, 3]}
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_json(path) == payload

    def test_missing_file_raises(self, tmp_path):
        """A nonexistent path must raise FileNotFoundError, not a generic OSError."""
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "does_not_exist.json")

    def test_accepts_str_path(self, tmp_path):
        """load_json must also accept a plain string path, not just pathlib.Path."""
        path = tmp_path / "data.json"
        path.write_text("{}", encoding="utf-8")
        assert load_json(str(path)) == {}


class TestLoadParquet:
    def test_round_trip(self, tmp_path):
        """A DataFrame written to parquet must be read back with identical values."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df.to_parquet(tmp_path / "file.parquet", index=False)
        loaded = load_parquet("file.parquet", tmp_path)
        pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df)

    def test_extension_is_optional(self, tmp_path):
        """Passing a filename without the .parquet suffix must still resolve correctly."""
        df = pd.DataFrame({"a": [1]})
        df.to_parquet(tmp_path / "file.parquet", index=False)
        loaded = load_parquet("file", tmp_path)
        assert len(loaded) == 1

    def test_missing_file_raises(self, tmp_path):
        """A nonexistent parquet path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_parquet("missing", tmp_path)

    def test_empty_dataframe_raises(self, tmp_path):
        """An empty parquet file is almost always a pipeline bug upstream, so it must raise ValueError."""
        pd.DataFrame({"a": []}).to_parquet(tmp_path / "empty.parquet", index=False)
        with pytest.raises(ValueError):
            load_parquet("empty", tmp_path)

    def test_non_default_index_warns(self, tmp_path):
        """A non-RangeIndex DataFrame should surface a UserWarning (silent-bug guard)."""
        df = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
        df.to_parquet(tmp_path / "file.parquet", index=True)
        with pytest.warns(UserWarning):
            load_parquet("file", tmp_path)


class TestLoadSplit:
    def test_builds_expected_path_and_loads(self, tmp_path):
        """load_split must resolve {data_dir}/{dataset_name}/{split}_{split_tag}.parquet."""
        dataset_dir = tmp_path / "my_dataset"
        dataset_dir.mkdir()
        df = pd.DataFrame({"a": [1, 2]})
        df.to_parquet(dataset_dir / "train_default.parquet", index=False)

        loaded = load_split(tmp_path, "my_dataset", "train", split_tag="default")
        assert len(loaded) == 2
