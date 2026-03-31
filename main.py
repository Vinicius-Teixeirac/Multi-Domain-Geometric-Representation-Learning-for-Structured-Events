# main.py
from pathlib import Path
import argparse
import yaml
import hashlib
import json

# ---------------------------------------------------------------------
# IMPORTANT: importing paths guarantees folder creation (side-effect)
# ---------------------------------------------------------------------
from src.config import paths  # noqa: F401

# ---------------------------------------------------------------------
# Runners (single-responsibility stages)
# ---------------------------------------------------------------------
from src.runners.cleaning_runner import ensure_cleaned
from src.runners.splitting_runner import ensure_splits
from src.runners.entity_runner import ensure_entities
from src.runners.tabular_runner import ensure_tabular_features
from src.runners.text_runner import ensure_text

from src.runners.mlp_runner import run_mlp
from src.runners.gnn_runner import run_gnn
from src.runners.bert_runner import run_bert
from src.runners.multiview_runner import run_multiview

from src.utils.seed import set_seed
from src.utils.experiments_logging import get_logger
from src.utils.loading import load_json

# ---------------------------------------------------------------------
# CAMEO dictionaries (explicit + centralized)
# ---------------------------------------------------------------------
CAMEO_JSON_PATH = Path("notebooks/cameo_codes.json")


# --------------------------------------------------
# Load CAMEO dictionaries (JSON-backed, lightweight)
# --------------------------------------------------
cameo_data = load_json(Path("notebooks/cameo_codes.json"))

CAMEO_DICTIONARIES = {
    "EventCode": cameo_data["EVENT_CODES"],
    "Actor1Geo_CountryCode": cameo_data["GEO_COUNTRY_CODES"],
    "Actor2Geo_CountryCode": cameo_data["GEO_COUNTRY_CODES"],
    "ActionGeo_CountryCode": cameo_data["GEO_COUNTRY_CODES"],
    "Actor1Geo_FeatureID": cameo_data["FEATURE_ID_CODES"],
    "Actor2Geo_FeatureID": cameo_data["FEATURE_ID_CODES"],
    "ActionGeo_FeatureID": cameo_data["FEATURE_ID_CODES"],
    "Actor1CountryCode": cameo_data["ACTOR_COUNTRY_CODES"],
    "Actor2CountryCode": cameo_data["ACTOR_COUNTRY_CODES"],
    "Actor1EthnicCode": cameo_data["ACTOR_ETHNIC_CODES"],
    "Actor2EthnicCode": cameo_data["ACTOR_ETHNIC_CODES"],
    "Actor1Type1Code": cameo_data["ACTOR_TYPE_SHORT"],
    "Actor1Type2Code": cameo_data["ACTOR_TYPE_SHORT"],
    "Actor1Type3Code": cameo_data["ACTOR_TYPE_SHORT"],
    "Actor2Type1Code": cameo_data["ACTOR_TYPE_SHORT"],
    "Actor2Type2Code": cameo_data["ACTOR_TYPE_SHORT"],
    "Actor2Type3Code": cameo_data["ACTOR_TYPE_SHORT"],
    "Actor1Religion1Code": cameo_data["ACTOR_RELIGION_CODES"],
    "Actor1Religion2Code": cameo_data["ACTOR_RELIGION_CODES"],
    "Actor2Religion1Code": cameo_data["ACTOR_RELIGION_CODES"],
    "Actor2Religion2Code": cameo_data["ACTOR_RELIGION_CODES"],
    "Actor1KnownGroupCode": cameo_data["ACTOR_KNOWN_GROUP_CODES"],
    "Actor2KnownGroupCode": cameo_data["ACTOR_KNOWN_GROUP_CODES"],
}

logger = get_logger("MAIN")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def hash_config(cfg: dict) -> str:
    """Stable hash for experiment identity."""
    cfg_for_hash = {
        k: v for k, v in cfg.items()
        if k not in {"_config_path"}
    }
    raw = json.dumps(cfg_for_hash, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:8]


def discover_datasets(raw_root: Path):
    return sorted(
        p.stem for p in raw_root.iterdir()
        if p.is_file() and p.suffix == ".parquet"
    )



def load_configs(config_paths):
    """Load multiple YAML experiment configs."""
    configs = []
    for path in config_paths:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg["_config_path"] = str(path)
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("GDELT full experimental runner")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Dataset names to process (stems of .parquet files in data/raw/). "
             "Defaults to all discovered datasets.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all stages (ignore cached artifacts)",
    )

    parser.add_argument(
        "--mlp-configs",
        type=str,
        nargs="*",
        default=[],
        help="List of MLP YAML config files",
    )

    parser.add_argument(
        "--gnn-configs",
        type=str,
        nargs="*",
        default=[],
        help="List of GNN YAML config files (homo or hetero)",
    )

    parser.add_argument(
        "--bert-configs",
        type=str,
        nargs="*",
        default=[],
        help="List of BERT YAML config files",
    )

    parser.add_argument(
        "--multiview-configs",
        type=str,
        nargs="*",
        default=[],
        help="List of multiview geometric model YAML config files",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    all_datasets = discover_datasets(paths.RAW_DATA)
    if args.datasets:
        unknown = set(args.datasets) - set(all_datasets)
        if unknown:
            raise ValueError(f"Unknown datasets (not found in data/raw/): {sorted(unknown)}")
        datasets = [d for d in all_datasets if d in set(args.datasets)]
    else:
        datasets = all_datasets
    logger.info(f"Datasets to process: {datasets}")

    mlp_configs       = load_configs(args.mlp_configs)
    gnn_configs       = load_configs(args.gnn_configs)
    bert_configs      = load_configs(args.bert_configs)
    multiview_configs = load_configs(args.multiview_configs)

    for dataset in datasets:
        logger.info("=" * 80)
        logger.info(f"DATASET: {dataset}")
        logger.info("=" * 80)

        # --------------------------------------------------
        # Stage 1: cleaning
        # --------------------------------------------------
        ensure_cleaned(sample_name=dataset, force=args.force, target_cols=['QuadClass'])

        # --------------------------------------------------
        # Stage 2: splits (split-tag aware, seed-namespaced)
        # --------------------------------------------------
        for base_cfg in (mlp_configs + gnn_configs + bert_configs + multiview_configs):
            base_tag = (
                base_cfg.get("data", {})
                .get("split", {})
                .get("tag", "default")
            )
            split_tag = f"{base_tag}_s{args.seed}"
            ensure_splits(sample_name=dataset, cleaned_filename=f"processed_{dataset}", stratify_by="QuadClass", tag=split_tag, random_state=args.seed, force=args.force)

        # --------------------------------------------------
        # Stage 3–5: entities, tabular features, text
        # --------------------------------------------------
        for base_cfg in (mlp_configs + gnn_configs + bert_configs):
            base_tag = (
                base_cfg.get("data", {})
                .get("split", {})
                .get("tag", "default")
            )
            split_tag = f"{base_tag}_s{args.seed}"

            ensure_entities(dataset, split_tag=split_tag, force=args.force)
            ensure_tabular_features(dataset, split_tag=split_tag, force=args.force)

            # CAMEO → natural language (BERT-ready)
            ensure_text(
                dataset,
                split_tag=split_tag,
                dictionaries=CAMEO_DICTIONARIES,
                force=args.force,
            )

        # --------------------------------------------------
        # Stage 6a: MLP experiments
        # --------------------------------------------------
        for base_cfg in mlp_configs:
            base_tag = (
                base_cfg.get("data", {})
                .get("split", {})
                .get("tag", "default")
            )
            split_tag = f"{base_tag}_s{args.seed}"

            cfg = {
                **base_cfg,
                "dataset": dataset,
                "seed": args.seed,
                "split_tag": split_tag,
            }

            exp_id = hash_config(cfg)
            cfg["exp_id"] = exp_id

            logger.info("-" * 60)
            logger.info(
                f"[MLP] Dataset={dataset} | "
                f"Split={split_tag} | "
                f"Config={base_cfg['_config_path']} | "
                f"Seed={args.seed} | "
                f"ID={exp_id}"
            )
            logger.info("-" * 60)

            run_mlp(cfg)

        # --------------------------------------------------
        # Stage 6b: GNN experiments
        # --------------------------------------------------
        for base_cfg in gnn_configs:
            base_tag = (
                base_cfg.get("data", {})
                .get("split", {})
                .get("tag", "default")
            )
            split_tag = f"{base_tag}_s{args.seed}"

            cfg = {
                **base_cfg,
                "dataset": dataset,
                "seed": args.seed,
                "split_tag": split_tag,
            }

            exp_id = hash_config(cfg)
            cfg["exp_id"] = exp_id

            graph_type = cfg["graph"]["type"]

            logger.info("-" * 60)
            logger.info(
                f"[GNN:{graph_type.upper()}] Dataset={dataset} | "
                f"Split={split_tag} | "
                f"Config={base_cfg['_config_path']} | "
                f"Seed={args.seed} | "
                f"ID={exp_id}"
            )
            logger.info("-" * 60)

            run_gnn(cfg)

        # --------------------------------------------------
        # Stage 6c: BERT experiments
        # --------------------------------------------------
        for base_cfg in bert_configs:
            base_tag = (
                base_cfg.get("data", {})
                .get("split", {})
                .get("tag", "default")
            )
            split_tag = f"{base_tag}_s{args.seed}"

            cfg = {
                **base_cfg,
                "dataset": dataset,
                "seed": args.seed,
                "split_tag": split_tag,
                "data": {
                    **base_cfg.get("data", {}),
                    "dictionaries": CAMEO_DICTIONARIES,
                },
            }

            exp_id = hash_config(cfg)
            cfg["exp_id"] = exp_id

            logger.info("-" * 60)
            logger.info(
                f"[BERT] Dataset={dataset} | "
                f"Split={split_tag} | "
                f"Config={base_cfg['_config_path']} | "
                f"Seed={args.seed} | "
                f"ID={exp_id}"
            )
            logger.info("-" * 60)

            run_bert(cfg)

        # --------------------------------------------------
        # Stage 6d: Multiview Geometric experiments
        # --------------------------------------------------
        for base_cfg in multiview_configs:
            base_tag = (
                base_cfg.get("data", {})
                .get("split", {})
                .get("tag", "default")
            )
            split_tag = f"{base_tag}_s{args.seed}"

            logger.info("-" * 60)
            logger.info(
                f"[MULTIVIEW] Dataset={dataset} | "
                f"Split={split_tag} | "
                f"Config={base_cfg['_config_path']} | "
                f"Seed={args.seed}"
            )
            logger.info("-" * 60)

            run_multiview(
                cfg=base_cfg,
                dataset_name=dataset,
                split_tag=split_tag,
                seed=args.seed,
            )

    logger.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
