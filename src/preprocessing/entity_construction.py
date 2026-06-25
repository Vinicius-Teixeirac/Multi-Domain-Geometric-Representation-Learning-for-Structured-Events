# src/preprocessing/entity_construction.py

from pathlib import Path
from typing import List, Dict

import pandas as pd

from src.config.paths import SPLITS_DATA, ENTITIES_DATA
from src.utils.loading import load_parquet
from src.utils.experiments_logging import get_logger
from src.utils.constants import NULL_TOKEN

logger = get_logger(__name__)

__all__ = ["build_event_entities"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_str(val) -> str:
    """Return str(val), or empty string for NaN/None."""
    if pd.isna(val):
        return ""
    return str(val)


def _safe_float(val, decimals: int = 3) -> str:
    """Return val rounded to `decimals` places as a string, or empty string for NaN/None."""
    if pd.isna(val):
        return ""
    return f"{round(float(val), decimals)}"


def _concat_non_null(values: List[str]) -> str:
    """Join non-empty, non-null tokens with '-' to form a composite entity ID."""
    non_null = [v for v in values if v not in {"", NULL_TOKEN}]
    return "-".join(non_null)


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
def build_event_entities(
    dataset_name: str,
    split_tag: str = "default",
) -> None:
    """
    Builds graph-only entity IDs *per split* (event-inductive).

    Also saves metadata with entity cardinalities for validation.
    """

    in_dir = SPLITS_DATA / dataset_name
    out_dir = ENTITIES_DATA / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        in_path = in_dir / f"{split}_{split_tag}.parquet"
        out_path = out_dir / f"{split}_{split_tag}_entities.parquet"
        meta_path = out_dir / f"{split}_{split_tag}_metadata.json"

        if not in_path.exists():
            logger.warning(f"Missing split: {in_path}")
            continue

        logger.info(f"Constructing entities [{dataset_name} | {split}]")

        df = load_parquet(f"{split}_{split_tag}.parquet", in_dir)

        # --------------------------------------------------
        # Actor 1
        # --------------------------------------------------
        actor1_cols = [
            "Actor1Name",
            "Actor1CountryCode",
            "Actor1KnownGroupCode",
            "Actor1EthnicCode",
            "Actor1Religion1Code",
            "Actor1Religion2Code",
            "Actor1Type1Code",
            "Actor1Type2Code",
            "Actor1Type3Code",
        ]

        df["Actor1ID"] = df[actor1_cols].apply(
            lambda r: _concat_non_null([_safe_str(r[c]) for c in actor1_cols]),
            axis=1,
        )

        # --------------------------------------------------
        # Actor 2
        # --------------------------------------------------
        actor2_cols = [
            "Actor2Name",
            "Actor2CountryCode",
            "Actor2KnownGroupCode",
            "Actor2EthnicCode",
            "Actor2Religion1Code",
            "Actor2Religion2Code",
            "Actor2Type1Code",
            "Actor2Type2Code",
            "Actor2Type3Code",
        ]

        df["Actor2ID"] = df[actor2_cols].apply(
            lambda r: _concat_non_null([_safe_str(r[c]) for c in actor2_cols]),
            axis=1,
        )

        # --------------------------------------------------
        # Event Geo (rounded floats)
        # --------------------------------------------------
        geo_cols = [
            "Actor1Geo_FeatureID",
            "Actor1Geo_Lat",
            "Actor1Geo_Long",
            "Actor2Geo_FeatureID",
            "Actor2Geo_Lat",
            "Actor2Geo_Long",
            "ActionGeo_FeatureID",
            "ActionGeo_Lat",
            "ActionGeo_Long",
        ]

        def build_geo_id(row) -> str:
            """Compose a geo entity ID from FeatureIDs and rounded lat/lon coordinates."""
            parts = []
            for c in geo_cols:
                if "Lat" in c or "Long" in c:
                    parts.append(_safe_float(row[c], decimals=3))
                else:
                    parts.append(_safe_str(row[c]))
            return _concat_non_null(parts)

        df["Event_GeoID"] = df.apply(build_geo_id, axis=1)

        df = df[["GlobalEventID","Actor1ID","Actor2ID","Event_GeoID","Day","QuadClass"]].copy()
        # --------------------------------------------------
        # Save entities
        # --------------------------------------------------
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved entities -> {out_path}")

        # --------------------------------------------------
        # Metadata (entity cardinalities)
        # --------------------------------------------------
        metadata: Dict[str, int] = {
            "num_actor1_entities": df["Actor1ID"].nunique(),
            "num_actor2_entities": df["Actor2ID"].nunique(),
            "num_geo_entities": df["Event_GeoID"].nunique(),
            "num_events": len(df),
        }

        pd.Series(metadata).to_json(meta_path, indent=2)
        logger.info(f"Saved entity metadata -> {meta_path}")
