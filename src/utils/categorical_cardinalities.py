"""Reads fitted categorical encoder artifacts to recover embedding vocab sizes.

Used wherever a model needs to size its embedding layers (nn.Embedding) to
match the vocabulary a SafeLabelEncoder/HashEncoder was fit with during the
tabular pipeline stage.
"""

from pathlib import Path
from typing import Dict

from src.config.schema.encoding_schema import ENCODING_SCHEMA
from src.representation.tabular.encoding import SafeLabelEncoder, HashEncoder

def load_categorical_cardinalities(
    categorical_cols: list,
    artifacts_dir: Path,
) -> Dict[str, int]:
    """
    Read fitted encoder artifacts and return per-column embedding vocab sizes.

    Parameters
    ----------
    categorical_cols : list[str]
        Column names to resolve cardinalities for.
    artifacts_dir : Path
        Directory containing `{column}.json` encoder artifacts saved by
        TabularPipeline.

    Returns
    -------
    dict[str, int]
        Mapping from column name to vocabulary size (num_classes for label
        encoding, num_buckets for hash encoding). Columns with no entry in
        ENCODING_SCHEMA or no artifact file on disk are silently skipped
        rather than raising, since callers only need cardinalities for the
        columns actually present in a given feature set.
    """
    cardinalities: Dict[str, int] = {}

    for col in categorical_cols:
        enc_cfg = ENCODING_SCHEMA.get(col)
        if enc_cfg is None:
            continue

        path = artifacts_dir / f"{col}.json"
        if not path.exists():
            continue

        method = enc_cfg["method"]

        if method == "label":
            enc = SafeLabelEncoder.load(path)
            cardinalities[col] = enc.num_classes_
        elif method == "hash":
            enc = HashEncoder.load(path)
            cardinalities[col] = enc.num_buckets

    return cardinalities