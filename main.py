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

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["mlp", "gnn", "bert"])
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    datasets = discover_datasets(paths.RAW_DATA)

    for dataset in datasets:

        logger.info("=" * 80)
        logger.info(f"DATASET: {dataset}")
        logger.info("=" * 80)

        ensure_cleaned(
            sample_name=dataset,
            force=args.force,
            target_cols=['QuadClass']
        )
        split_tag = f"default_s{args.seed}"

        ensure_splits(
            sample_name=dataset,
            cleaned_filename=f"processed_{dataset}",
            stratify_by="QuadClass",
            tag=split_tag,
            random_state=args.seed,
            force=args.force
        )

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

    logger.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()