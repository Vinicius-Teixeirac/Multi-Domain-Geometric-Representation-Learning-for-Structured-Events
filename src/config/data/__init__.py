"""Compatibility shim for `src.config.data`.

This package provides a small, derived view of the project's column
constants so older imports (e.g. `src.config.data.columns`) keep working.
"""

from . import columns  # re-export module
from . import column_type  # re-export module

__all__ = ["columns", "column_type"]
