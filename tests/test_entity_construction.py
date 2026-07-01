"""End-to-end tests for build_event_entities against real split parquets."""

import json

import pandas as pd
import pytest

import src.preprocessing.entity_construction as ec_mod
from src.preprocessing.entity_construction import build_event_entities


def _make_split_df(n: int) -> pd.DataFrame:
    """Minimal split DataFrame with every column build_event_entities reads."""
    return pd.DataFrame({
        "GlobalEventID": range(n),
        "Actor1Name": [f"actor1_{i % 3}" for i in range(n)],
        "Actor1CountryCode": ["USA"] * n,
        "Actor1KnownGroupCode": [None] * n,
        "Actor1EthnicCode": [None] * n,
        "Actor1Religion1Code": [None] * n,
        "Actor1Religion2Code": [None] * n,
        "Actor1Type1Code": [None] * n,
        "Actor1Type2Code": [None] * n,
        "Actor1Type3Code": [None] * n,
        "Actor2Name": [f"actor2_{i % 2}" for i in range(n)],
        "Actor2CountryCode": ["CAN"] * n,
        "Actor2KnownGroupCode": [None] * n,
        "Actor2EthnicCode": [None] * n,
        "Actor2Religion1Code": [None] * n,
        "Actor2Religion2Code": [None] * n,
        "Actor2Type1Code": [None] * n,
        "Actor2Type2Code": [None] * n,
        "Actor2Type3Code": [None] * n,
        "Actor1Geo_FeatureID": ["G1"] * n,
        "Actor1Geo_Lat": [10.0] * n,
        "Actor1Geo_Long": [20.0] * n,
        "Actor2Geo_FeatureID": ["G2"] * n,
        "Actor2Geo_Lat": [30.0] * n,
        "Actor2Geo_Long": [40.0] * n,
        "ActionGeo_FeatureID": ["G3"] * n,
        "ActionGeo_Lat": [50.0] * n,
        "ActionGeo_Long": [60.0] * n,
        "Day": [20200101 + i for i in range(n)],
        "QuadClass": [i % 4 for i in range(n)],
    })


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    """Redirect SPLITS_DATA/ENTITIES_DATA to a temp directory for isolated tests."""
    splits_dir = tmp_path / "splits"
    entities_dir = tmp_path / "entities"
    monkeypatch.setattr(ec_mod, "SPLITS_DATA", splits_dir)
    monkeypatch.setattr(ec_mod, "ENTITIES_DATA", entities_dir)
    return splits_dir, entities_dir


class TestBuildEventEntities:
    def test_builds_entities_for_available_splits(self, patched_dirs):
        """train/test present, valid absent -> only train/test entity files are written."""
        splits_dir, entities_dir = patched_dirs
        dataset_dir = splits_dir / "my_dataset"
        dataset_dir.mkdir(parents=True)
        _make_split_df(10).to_parquet(dataset_dir / "train_default.parquet", index=False)
        _make_split_df(5).to_parquet(dataset_dir / "test_default.parquet", index=False)

        build_event_entities("my_dataset", split_tag="default")

        out_dir = entities_dir / "my_dataset"
        assert (out_dir / "train_default_entities.parquet").exists()
        assert (out_dir / "test_default_entities.parquet").exists()
        assert not (out_dir / "valid_default_entities.parquet").exists()

    def test_output_columns_and_composite_ids(self, patched_dirs):
        """Output must expose the expected columns, with Actor1ID built from actor1 attributes."""
        splits_dir, entities_dir = patched_dirs
        dataset_dir = splits_dir / "my_dataset"
        dataset_dir.mkdir(parents=True)
        _make_split_df(4).to_parquet(dataset_dir / "train_default.parquet", index=False)

        build_event_entities("my_dataset", split_tag="default")

        out = pd.read_parquet(entities_dir / "my_dataset" / "train_default_entities.parquet")
        assert set(out.columns) == {
            "GlobalEventID", "Actor1ID", "Actor2ID", "Event_GeoID", "Day", "QuadClass",
        }
        assert out["Actor1ID"].str.startswith("actor1_").all()

    def test_metadata_cardinalities_are_correct(self, patched_dirs):
        """Metadata JSON must report the correct number of unique actor1/actor2/geo entities."""
        splits_dir, entities_dir = patched_dirs
        dataset_dir = splits_dir / "my_dataset"
        dataset_dir.mkdir(parents=True)
        _make_split_df(6).to_parquet(dataset_dir / "train_default.parquet", index=False)

        build_event_entities("my_dataset", split_tag="default")

        meta = json.loads((entities_dir / "my_dataset" / "train_default_metadata.json").read_text())
        assert meta["num_events"] == 6
        assert meta["num_actor1_entities"] == 3  # actor1_0, actor1_1, actor1_2
        assert meta["num_actor2_entities"] == 2  # actor2_0, actor2_1

    def test_missing_split_is_skipped_without_raising(self, patched_dirs):
        """A dataset with no split files at all must complete without raising (all splits skipped)."""
        splits_dir, entities_dir = patched_dirs
        dataset_dir = splits_dir / "my_dataset"
        dataset_dir.mkdir(parents=True)
        # no split parquets written at all

        build_event_entities("my_dataset", split_tag="default")

        out_dir = entities_dir / "my_dataset"
        assert not any(out_dir.iterdir()) if out_dir.exists() else True
