# src/runners/cleaning_runner.py
"""
Runner for the data-cleaning pipeline stage.

Wraps DataCleaner with an idempotency check: if a cleaned parquet already
exists for a given sample, cleaning is skipped unless ``force=True``. This
is the first stage in main.py's per-dataset pipeline, upstream of
splitting, entity construction, tabular/text feature building, and model
training.
"""

from pathlib import Path
from typing import Iterable, Optional, List

from src.preprocessing.cleaning import DataCleaner
from src.config.paths import PROCESSED_DATA
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)


def ensure_cleaned(
    *,
    sample_name: str,
    columns: Optional[Iterable[str]] = None,
    target_cols: Optional[List[str]] = None,
    force: bool = False,
) -> Path:
    """
    Ensure that a cleaned dataset exists.

    Parameters
    ----------
    sample_name : str
        Name of the raw sample (stem of the parquet file in data/raw/).
    columns : Iterable[str], optional
        Subset of columns to keep during cleaning; None keeps all
        recognized columns.
    target_cols : List[str], optional
        Label columns to normalize (e.g. remapping QuadClass 1-4 to 0-3).
    force : bool
        If True, re-run cleaning even if the cleaned parquet already
        exists, overwriting it. If False (default), an existing cleaned
        file is reused as-is.

    Returns
    -------
    Path
        Path to the cleaned parquet file.
    """
    output_dir = PROCESSED_DATA / sample_name
    output_path = output_dir / f"processed_{sample_name}.parquet"

    if output_path.exists() and not force:
        logger.info(f"Cleaned file already exists, skipping: {output_path}")
        return output_path

    logger.info(f"Running cleaning for sample '{sample_name}'")

    cleaner = DataCleaner(
        sample_name=sample_name,
        columns=columns,
    )

    return cleaner.run(
        sample_name=sample_name,
        target_cols=target_cols,
    )
