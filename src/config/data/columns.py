"""Derived column list for GDELT event data.

This file exports `GDELT_EVENT_COLUMNS` expected by the preprocessing
code. It derives the list from the authoritative `COLUMNS_SCHEMA` so
there's only one source of truth.
"""
from typing import List

from src.config.schema.columns_schema import COLUMNS_SCHEMA


GDELT_EVENT_COLUMNS: List[str] = list(COLUMNS_SCHEMA.keys())

__all__ = ["GDELT_EVENT_COLUMNS"]
