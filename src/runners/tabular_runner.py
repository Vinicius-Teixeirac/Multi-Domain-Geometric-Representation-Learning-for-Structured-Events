# src/runners/tabular_runner.py
"""
Runner for the tabular feature-engineering pipeline stage.

Wraps TabularPipeline with an idempotency check: feature parquet files
are only (re)built when missing or when ``force=True``. Features
produced here are consumed by the MLP and (when node_features="all")
GNN runners.
"""

from src.config.paths import FEATURES_DATA
from src.representation.tabular.tabular_pipeline import TabularPipeline
from src.utils.experiments_logging import get_logger

logger = get_logger(__name__)


def ensure_tabular_features(
    dataset: str,
    *,
    split_tag: str = "default",
    force: bool = False,
) -> dict:
    """
    Ensure tabular features exist for a dataset and split tag.

    Artifacts:
        data/features/{dataset}/
            train_{tag}_features.parquet
            valid_{tag}_features.parquet (optional)
            test_{tag}_features.parquet

    Idempotent unless force=True.

    Parameters
    ----------
    dataset : str
        Dataset name (stem of the parquet file in data/raw/).
    split_tag : str
        Split identifier whose train/valid/test feature files are checked
        and (re)built.
    force : bool
        If True, rebuild tabular features even if the expected files
        already exist. If False (default), an existing complete set is
        reused.

    Returns
    -------
    dict
        ``{"skipped": bool, "dataset": str, "split_tag": str}`` indicating
        whether feature engineering was skipped due to existing artifacts.
    """

    logger.info(
        f"[TABULAR] Dataset={dataset} | split_tag={split_tag} | force={force}"
    )

    out_dir = FEATURES_DATA / dataset

    expected_train = out_dir / f"train_{split_tag}_features.parquet"
    expected_test = out_dir / f"test_{split_tag}_features.parquet"

    if expected_train.exists() and expected_test.exists() and not force:
        logger.info("Tabular features already exist - skipping")
        return {"skipped": True, "dataset": dataset, "split_tag": split_tag}

    logger.info("Running tabular feature pipeline...")
    pipeline = TabularPipeline(dataset_name=dataset, split_tag=split_tag)
    pipeline.run()
    return {"skipped": False, "dataset": dataset, "split_tag": split_tag}
