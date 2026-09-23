"""Tests for the column-to-dictionary mapping the event verbaliser resolves through.

A column missing from the mapping does not raise: translate_code returns None
for an empty dictionary and the caller drops the phrase, so the sentence comes
out shorter with nothing to indicate why. These tests pin the mapping so that
failure cannot return unnoticed.
"""

import pandas as pd

from main import CAMEO_DICTIONARIES
from src.config.schema.chosen_columns import CHOSEN_COLUMNS
from src.representation.text.text_builder import event_to_text, verbalize_event_location

# Every column text_builder translates, for both actors plus the event location.
TRANSLATED_COLUMNS = [
    f"{prefix}{suffix}"
    for prefix in ("Actor1", "Actor2")
    for suffix in ("Type1Code", "KnownGroupCode", "Religion1Code", "EthnicCode", "CountryCode")
] + ["Actor1Geo_FeatureID", "Actor2Geo_FeatureID", "ActionGeo_FeatureID"]


def _row(**overrides):
    base = {
        "Actor1Name": "obama", "Actor2Name": "american",
        "Actor1CountryCode": "USA", "Actor2CountryCode": "USA",
        "Actor1Type1Code": "GOV", "Actor2Type1Code": None,
        "Actor1KnownGroupCode": None, "Actor2KnownGroupCode": None,
        "Actor1Religion1Code": None, "Actor2Religion1Code": None,
        "Actor1EthnicCode": None, "Actor2EthnicCode": None,
        "Actor1Geo_FeatureID": "531871", "Actor2Geo_FeatureID": "531871",
        "ActionGeo_FeatureID": "531871", "Day": 20160923,
    }
    base.update(overrides)
    return pd.Series(base)


class TestMappingCompleteness:
    def test_every_translated_column_has_a_dictionary(self):
        missing = [c for c in TRANSLATED_COLUMNS if not CAMEO_DICTIONARIES.get(c)]
        assert not missing, f"columns resolving to nothing: {missing}"

    def test_no_dictionary_for_a_column_the_pipeline_drops(self):
        """EventCode was mapped for years while CHOSEN_COLUMNS never retained it."""
        assert "EventCode" not in CAMEO_DICTIONARIES
        assert all(column in CHOSEN_COLUMNS for column in CAMEO_DICTIONARIES)

    def test_dictionaries_are_populated(self):
        assert all(len(d) > 0 for d in CAMEO_DICTIONARIES.values())


class TestVerbalisedSentence:
    def test_the_where_phrase_is_present(self):
        text = event_to_text(_row(), CAMEO_DICTIONARIES, "interacted with")
        assert "[WHERE] Washington, District Of Columbia, United States." in text

    def test_actor_attributes_are_present(self):
        text = event_to_text(_row(), CAMEO_DICTIONARIES, "interacted with")
        assert "from United States" in text

    def test_an_empty_mapping_drops_the_location_without_raising(self):
        """The failure mode this module exists to catch."""
        text = event_to_text(_row(), {}, "interacted with")
        assert "[WHERE]" not in text and "[WHO]" in text

    def test_an_unknown_feature_id_yields_no_location(self):
        assert verbalize_event_location(_row(ActionGeo_FeatureID="0"), CAMEO_DICTIONARIES) is None
