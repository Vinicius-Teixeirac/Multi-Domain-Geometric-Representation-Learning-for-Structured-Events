# src/runners/cleaning_runner.py

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

    Returns
    -------
    Path
        Path to the cleaned parquet file.
    """
    output_dir = PROCESSED_DATA / sample_name
    output_path = output_dir / f"processed_{sample_name}"

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
