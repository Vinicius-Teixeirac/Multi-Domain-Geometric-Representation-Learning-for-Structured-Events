# main.py
from pathlib import Path
import argparse
import yaml
import hashlib
import json

from src.config import paths  # noqa: F401

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

logger = get_logger("MAIN")

CAMEO_JSON_PATH = Path("notebooks/cameo_codes.json")
cameo_data = load_json(CAMEO_JSON_PATH)

CAMEO_DICTIONARIES = {
    "EventCode": cameo_data["EVENT_CODES"],
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def hash_config(cfg: dict) -> str:
    raw = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:8]


def discover_datasets(raw_root: Path):
    return sorted(
        p.stem for p in raw_root.iterdir()
        if p.is_file() and p.suffix == ".parquet"
    )


def parse_args():
    parser = argparse.ArgumentParser("Single experiment runner")

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


# --------------------------------------------------
# Main
# --------------------------------------------------
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

        ensure_text(
            dataset,
            split_tag=split_tag,
            dictionaries=CAMEO_DICTIONARIES,
            force=args.force,
        )

        cfg = {
            **base_cfg,
            "dataset": dataset,
            "seed": args.seed,
            "split_tag": split_tag,
        }

        exp_id = hash_config(cfg)
        cfg["exp_id"] = exp_id

        logger.info(f"Running {args.model_type.upper()} | ID={exp_id}")

        if args.model_type == "mlp":
            run_mlp(cfg)
        elif args.model_type == "gnn":
            run_gnn(cfg)
        elif args.model_type == "bert":
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