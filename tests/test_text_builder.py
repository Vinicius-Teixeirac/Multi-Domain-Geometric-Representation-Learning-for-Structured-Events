"""Tests for verbalizing structured GDELT rows into natural-language sentences."""

import pandas as pd
import pytest

from src.representation.text.text_builder import (
    translate_code,
    format_day,
    normalize_name,
    verbalize_actor,
    verbalize_event_location,
    event_to_text,
    build_event_texts,
)


class TestTranslateCode:
    def test_known_code_translates(self):
        """A code present in the dictionary must return its human-readable label."""
        assert translate_code("USA", {"USA": "United States"}) == "United States"

    def test_null_token_returns_none(self):
        """The explicit '__NULL__' sentinel must translate to None, not a literal string."""
        assert translate_code("__NULL__", {"USA": "United States"}) is None

    def test_nan_returns_none(self):
        """A pandas NaN code must translate to None."""
        assert translate_code(float("nan"), {}) is None

    def test_unknown_code_returns_none(self):
        """A code missing from the dictionary must return None, not raise KeyError."""
        assert translate_code("ZZZ", {"USA": "United States"}) is None


class TestFormatDay:
    def test_yyyymmdd_format(self):
        """An 8-digit GDELT date must format as 'Month YYYY'."""
        assert format_day(20150923) == "September 2015"

    def test_yyyymm_format(self):
        """A 6-digit MonthYear value must also format as 'Month YYYY'."""
        assert format_day(201509) == "September 2015"

    def test_none_returns_none(self):
        """A missing day value must return None."""
        assert format_day(None) is None

    def test_invalid_length_returns_none(self):
        """A malformed day (wrong digit count) must return None, not raise."""
        assert format_day(123) is None


class TestNormalizeName:
    def test_title_cases_name(self):
        """Names must be title-cased for consistent display."""
        assert normalize_name("UNITED STATES") == "United States"
        assert normalize_name("john smith") == "John Smith"


class TestVerbalizeActor:
    def test_missing_name_returns_none(self):
        """An actor with no name (NaN or NULL token) must produce no phrase at all."""
        row = {"Actor1Name": "__NULL__"}
        assert verbalize_actor(row, "Actor1", {}) is None

    def test_name_only_actor(self):
        """An actor with just a name and no other attributes still produces a phrase with the name."""
        row = {"Actor1Name": "john smith"}
        phrase = verbalize_actor(row, "Actor1", {})
        assert "John Smith" in phrase

    def test_role_is_appended_in_parentheses(self):
        """A resolved Type1Code role must be appended in parentheses after the name."""
        row = {"Actor1Name": "john smith", "Actor1Type1Code": "GOV"}
        dictionaries = {"Actor1Type1Code": {"GOV": "Government"}}
        phrase = verbalize_actor(row, "Actor1", dictionaries)
        assert "(Government)" in phrase


class TestVerbalizeEventLocation:
    def test_known_location(self):
        """A resolved ActionGeo_FeatureID must produce a title-cased location string."""
        row = {"ActionGeo_FeatureID": "US"}
        out = verbalize_event_location(row, {"ActionGeo_FeatureID": {"US": "united states"}})
        assert out == "United States"

    def test_unknown_location_returns_none(self):
        """An unresolved location code must return None."""
        row = {"ActionGeo_FeatureID": "ZZ"}
        assert verbalize_event_location(row, {"ActionGeo_FeatureID": {}}) is None


class TestEventToText:
    def test_both_actors_present(self):
        """With both actors present, the sentence must contain [WHO] ... verb ... [WHOM]."""
        row = {"Actor1Name": "alice", "Actor2Name": "bob", "Day": None}
        text = event_to_text(row, {}, verb="interacted with")
        assert "[WHO]" in text and "[WHOM]" in text and "interacted with" in text

    def test_only_one_actor_present(self):
        """With only one actor present, the sentence must fall back to a generic involvement phrase."""
        row = {"Actor1Name": "alice", "Actor2Name": "__NULL__", "Day": None}
        text = event_to_text(row, {}, verb="interacted with")
        assert "[WHO]" in text
        assert "[WHOM]" not in text

    def test_neither_actor_present(self):
        """With no actors at all, no [WHO]/[WHOM] tags should appear."""
        row = {"Actor1Name": "__NULL__", "Actor2Name": "__NULL__", "Day": None}
        text = event_to_text(row, {}, verb="interacted with")
        assert "[WHO]" not in text and "[WHOM]" not in text


class TestBuildEventTexts:
    def test_one_sentence_per_row(self):
        """build_event_texts must produce exactly one string per input row, in order."""
        df = pd.DataFrame({
            "Actor1Name": ["alice", "bob"],
            "Actor2Name": ["__NULL__", "__NULL__"],
            "Day": [None, None],
        })
        texts = build_event_texts(df, {})
        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)

    def test_verbs_cycle_by_row_position(self):
        """Verbs must be assigned round-robin by row position, so the same event always
        gets the same verb across runs (deterministic, not tied to DataFrame index)."""
        df = pd.DataFrame({
            "Actor1Name": ["a1", "a2", "a3", "a4"],
            "Actor2Name": ["b1", "b2", "b3", "b4"],
            "Day": [None, None, None, None],
        })
        texts_a = build_event_texts(df, {})
        texts_b = build_event_texts(df.copy(), {})
        assert texts_a == texts_b
