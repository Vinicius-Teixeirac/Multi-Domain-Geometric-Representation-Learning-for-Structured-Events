"""Tests for the column-to-dictionary mapping the event verbaliser resolves through.

A column missing from the mapping does not raise: translate_code returns None
for an empty dictionary and the caller drops the phrase, so the sentence comes
out shorter with nothing to indicate why. These tests pin the mapping so that
failure cannot return unnoticed.
"""

import pandas as pd
import pytest

from main import CAMEO_DICTIONARIES, cameo_data
from src.config.schema.chosen_columns import CHOSEN_COLUMNS
from src.representation.text.text_builder import event_to_text, verbalize_event_location

# Every column text_builder translates, for both actors plus the event location.
TRANSLATED_COLUMNS = [
    f"{prefix}{suffix}"
    for prefix in ("Actor1", "Actor2")
    for suffix in (
        "Type1Code", "Type2Code", "Type3Code", "KnownGroupCode",
        "Religion1Code", "Religion2Code", "EthnicCode", "CountryCode",
    )
] + ["Actor1Geo_FeatureID", "Actor2Geo_FeatureID", "ActionGeo_FeatureID"]


def _row(**overrides):
    base = {
        "Actor1Name": "obama", "Actor2Name": "american",
        "Actor1CountryCode": "USA", "Actor2CountryCode": "USA",
        "Actor1Type1Code": "GOV", "Actor2Type1Code": None,
        "Actor1KnownGroupCode": None, "Actor2KnownGroupCode": None,
        "Actor1Type2Code": None, "Actor2Type2Code": None,
        "Actor1Type3Code": None, "Actor2Type3Code": None,
        "Actor1Religion1Code": None, "Actor2Religion1Code": None,
        "Actor1Religion2Code": None, "Actor2Religion2Code": None,
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


class TestCodeCoverage:
    """Defects the BRACIS text runs carried: stale tables, code case, dropped columns."""

    @pytest.mark.parametrize("code", ["PLO", "OIC", "OAU"])
    def test_known_groups_the_old_table_lacked_resolve(self, code):
        assert code in cameo_data["ACTOR_KNOWN_GROUP_CODES"]

    def test_a_lower_case_ethnic_code_is_named(self):
        """GDELT writes 99% of EthnicCode values in lower case; keys are upper case."""
        text = event_to_text(_row(Actor1EthnicCode="kur"), CAMEO_DICTIONARIES, "met")
        assert cameo_data["ACTOR_ETHNIC_CODES"]["KUR"] in text

    def test_the_second_and_third_types_are_named(self):
        row = _row(Actor1Type2Code="MIL", Actor1Type3Code="REB")
        text = event_to_text(row, CAMEO_DICTIONARIES, "met")
        for code in ("GOV", "MIL", "REB"):
            assert cameo_data["ACTOR_TYPE_CODES"][code] in text

    def test_the_second_religion_is_named(self):
        row = _row(Actor1Religion1Code="CHR", Actor1Religion2Code="CTH")
        text = event_to_text(row, CAMEO_DICTIONARIES, "met")
        assert cameo_data["ACTOR_RELIGION_CODES"]["CTH"].lower() in text

    def test_a_label_repeated_across_codes_is_written_once(self):
        row = _row(Actor1Type2Code="GOV")
        text = event_to_text(row, CAMEO_DICTIONARIES, "met")
        assert text.count(cameo_data["ACTOR_TYPE_CODES"]["GOV"]) == 1

    def test_every_actor_key_is_upper_case(self):
        """translate_code's upper-case fallback is only complete if no key is lower case."""
        for name in ("ACTOR_ETHNIC_CODES", "ACTOR_TYPE_CODES", "ACTOR_RELIGION_CODES",
                     "ACTOR_KNOWN_GROUP_CODES", "ACTOR_COUNTRY_CODES"):
            assert all(k == k.upper() for k in cameo_data[name]), name
