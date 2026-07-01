# src/runners/entity_runner.py
"""
Runner for the graph-entity-ID construction pipeline stage.

Wraps build_event_entities with an idempotency check: entity ID parquet
files are only (re)built when missing or when ``force=True``. Entity IDs
are consumed downstream by the GNN and multi-domain graph representations.
"""

from src.config.paths import ENTITIES_DATA
from src.preprocessing.entity_construction import build_event_entities
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)


def ensure_entities(
    dataset: str,
    split_tag: str = "default",
    force: bool = False,
) -> dict:
    """
    Ensure graph entity IDs exist for a dataset.

    This is a wrapper around build_event_entities with idempotency.

    Parameters
    ----------
    dataset : str
        Dataset name (stem of the parquet file in data/raw/).
    split_tag : str
        Split identifier whose train/valid/test entity files are checked
        and (re)built.
    force : bool
        If True, rebuild entity IDs even if the expected files already
        exist. If False (default), an existing complete set is reused.

    Returns
    -------
    dict
        ``{"skipped": bool, "dataset": str, "split_tag": str}`` indicating
        whether construction was skipped due to existing artifacts.
    """

    logger.info(f"[ENTITIES] Dataset={dataset} | tag={split_tag}")

    out_dir = ENTITIES_DATA / dataset
    expected = [
        out_dir / f"train_{split_tag}_entities.parquet",
        out_dir / f"test_{split_tag}_entities.parquet",
    ]

    # Validation split is optional
    valid_path = out_dir / f"valid_{split_tag}_entities.parquet"
    if valid_path.exists():
        expected.append(valid_path)

    if all(p.exists() for p in expected) and not force:
        logger.info("Skipping entity construction (already exists)")
        return {"skipped": True, "dataset": dataset, "split_tag": split_tag}

    build_event_entities(dataset_name=dataset, split_tag=split_tag)
    return {"skipped": False, "dataset": dataset, "split_tag": split_tag}
