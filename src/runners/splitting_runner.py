# src/runners/splitting_runner.py

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
    test_size: float = 0.15,
    valid_size: Optional[float] = 0.10,
    stratify_by: Optional[str] = None,
    sort_by_time: Optional[str] = None,
    random_state: int,
    force: bool = False,
) -> None:
    """
    Ensure that dataset splits exist.

    Splits created:
    - train_{tag}.parquet
    - valid_{tag}.parquet (optional)
    - test_{tag}.parquet
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
        test_size=test_size,
        valid_size=valid_size,
        stratify_by=stratify_by,
        random_state=random_state,
        tag=tag,
    )
    return {"skipped": False, "sample_name": sample_name, "tag": tag}
