# src/runners/splitting_runner.py
"""
Runner for the train/valid/test splitting pipeline stage.

Wraps Splitter with an idempotency check: split parquet files are only
(re)built when missing or when ``force=True``. Splits produced here are
consumed by every downstream stage (entity construction, tabular/text
feature building, and all model runners).
"""

from pathlib import Path
from typing import Optional

from src.preprocessing.splitting import Splitter
from src.config.paths import SPLITS_DATA
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)


def ensure_splits(
    *,
    sample_name: str,
    cleaned_filename: str,
    tag: str = "default",
    # Standard 70/15/15 train/valid/test split proportions used across
    # this project's experiments.
    test_size: float = 0.15,
    valid_size: Optional[float] = 0.10,
    stratify_by: Optional[str] = None,
    sort_by_time: Optional[str] = None,
    random_state: int,
    force: bool = False,
) -> dict:
    """
    Ensure that dataset splits exist.

    Splits created:
    - train_{tag}.parquet
    - valid_{tag}.parquet (optional)
    - test_{tag}.parquet

    Parameters
    ----------
    sample_name : str
        Name of the raw sample (stem of the parquet file in data/raw/).
    cleaned_filename : str
        Stem of the cleaned parquet file (produced by ensure_cleaned) to
        split.
    tag : str
        Split identifier used to namespace the output filenames.
    test_size : float
        Fraction of rows held out for the test split.
    valid_size : float, optional
        Fraction of rows held out for the validation split; if None, no
        validation split is produced.
    stratify_by : str, optional
        Column to stratify the split on (e.g. "QuadClass") so class
        proportions are preserved across train/valid/test.
    sort_by_time : str, optional
        Column to sort by before splitting. When provided, ``splitter.run``
        is invoked with ``strategy="temporal"`` and this column as
        ``time_column`` (a chronological, non-shuffled split); when None,
        ``strategy="random"`` is used instead.
    random_state : int
        Seed controlling the random split (ignored for the temporal
        strategy's ordering, but still passed through).
    force : bool
        If True, rebuild splits even if the expected files already exist.
        If False (default), an existing complete set is reused.

    Returns
    -------
    dict
        ``{"skipped": bool, "sample_name": str, "tag": str}`` indicating
        whether splitting was skipped due to existing artifacts.
    """
    split_dir = SPLITS_DATA / sample_name

    expected_files = [
        split_dir / f"train_{tag}.parquet",
        split_dir / f"test_{tag}.parquet",
    ]
    if valid_size is not None:
        expected_files.append(split_dir / f"valid_{tag}.parquet")

    if all(p.exists() for p in expected_files) and not force:
        logger.info(f"Splits already exist for sample '{sample_name}', tag='{tag}'. Skipping.")
        return {"skipped": True, "sample_name": sample_name, "tag": tag}

    logger.info(f"Running splitting for sample '{sample_name}', tag='{tag}'")

    splitter = Splitter(sample_name=sample_name)

    splitter.run(
        filename=cleaned_filename,
        strategy="temporal" if sort_by_time else "random",
        time_column=sort_by_time,
        test_size=test_size,
        valid_size=valid_size,
        stratify_by=stratify_by,
        random_state=random_state,
        tag=tag,
    )
    return {"skipped": False, "sample_name": sample_name, "tag": tag}
