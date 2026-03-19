# src/runners/entity_runner.py

from src.config.paths import ENTITIES_DATA
from src.preprocessing.entity_construction import build_event_entities
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)


def ensure_entities(
    dataset: str,
    split_tag: str = "default",
    force: bool = False,
) -> None:
    """
    Ensure graph entity IDs exist for a dataset.

    This is a wrapper around build_event_entities with idempotency.
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
        return

    build_event_entities(dataset_name=dataset, split_tag=split_tag)
