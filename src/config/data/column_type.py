"""Column type groups used during cleaning/casting.

Derives integer/float/string groups from `COLUMNS_SCHEMA` to keep a
single source of truth for column semantics.
"""
from typing import List

from src.config.schema.columns_schema import COLUMNS_SCHEMA


INTEGER_COLUMNS: List[str] = [
    name for name, meta in COLUMNS_SCHEMA.items()
    if meta.get("kind") in ("id", "target")
]

FLOAT_COLUMNS: List[str] = [
    name for name, meta in COLUMNS_SCHEMA.items()
    if meta.get("kind") == "geo"
]

STRING_COLUMNS: List[str] = [
    name for name, meta in COLUMNS_SCHEMA.items()
    if meta.get("kind") == "categorical"
]

__all__ = ["INTEGER_COLUMNS", "FLOAT_COLUMNS", "STRING_COLUMNS"]
