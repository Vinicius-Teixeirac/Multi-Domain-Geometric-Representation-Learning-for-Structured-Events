# src/utils/categorical_cardinalities.py
from pathlib import Path
from typing import Dict

from src.config.schema.encoding_schema import ENCODING_SCHEMA
from src.representation.tabular.encoding import SafeLabelEncoder, HashEncoder

def load_categorical_cardinalities(
    categorical_cols: list,
    artifacts_dir: Path,
) -> Dict[str, int]:
    """Read fitted encoder artifacts and return {column: num_buckets_or_classes} for embedding layers."""
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