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
from src.runners.multi_domain_runner import run_multi_domain

from src.utils.seed import set_seed
from src.utils.experiments_logging import get_logger
from src.utils.loading import load_json

logger = get_logger("MAIN")

CAMEO_JSON_PATH = Path(__file__).resolve().parent / "notebooks" / "cameo_codes.json"
cameo_data = load_json(CAMEO_JSON_PATH)

CAMEO_DICTIONARIES = {
    "EventCode": cameo_data["EVENT_CODES"],
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_configs(paths_list: list[str]) -> list[dict]:
    """Load YAML config files and inject _config_path into each dict."""
    configs = []
    for path in (paths_list or []):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg["_config_path"] = path
        configs.append(cfg)
    return configs


def hash_config(cfg: dict) -> str:
    hashable = {k: v for k, v in cfg.items() if k != "_config_path"}
    raw = json.dumps(hashable, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:8]


def discover_datasets(raw_root: Path):
    return sorted(
        p.stem for p in raw_root.iterdir()
        if p.is_file() and p.suffix == ".parquet"
    )


def _split_tag(base_cfg: dict, seed: int) -> str:
    base = base_cfg.get("data", {}).get("split", {}).get("tag", "default")
    return f"{base}_s{seed}"


def parse_args():
    parser = argparse.ArgumentParser("GDELT experiment runner")

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
        "--multi-domain-configs",
        type=str,
        nargs="*",
        default=[],
        help="List of multi-domain geometric model YAML config files",
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

    mlp_configs          = load_configs(args.mlp_configs)
    gnn_configs          = load_configs(args.gnn_configs)
    bert_configs         = load_configs(args.bert_configs)
    multi_domain_configs = load_configs(args.multi_domain_configs)

    all_configs     = mlp_configs + gnn_configs + bert_configs + multi_domain_configs
    tabular_configs = mlp_configs + gnn_configs + bert_configs

    for dataset in datasets:

        logger.info("=" * 80)
        logger.info(f"DATASET: {dataset}")
        logger.info("=" * 80)

        # --------------------------------------------------
        # Stage 1: cleaning
        # --------------------------------------------------
        ensure_cleaned(sample_name=dataset, force=args.force, target_cols=["QuadClass"])

        # --------------------------------------------------
        # Stage 2: splits  (deduplicated by split_tag)
        # --------------------------------------------------
        seen_splits: set[str] = set()
        for base_cfg in all_configs:
            tag = _split_tag(base_cfg, args.seed)
            if tag not in seen_splits:
                ensure_splits(
                    sample_name=dataset,
                    cleaned_filename=f"processed_{dataset}",
                    stratify_by="QuadClass",
                    tag=tag,
                    random_state=args.seed,
                    force=args.force,
                )
                seen_splits.add(tag)

        # --------------------------------------------------
        # Stage 3–5: entities, tabular features, text
        #   (MLP / GNN / BERT only — multi-domain reads raw splits)
        # --------------------------------------------------
        seen_tabular: set[str] = set()
        for base_cfg in tabular_configs:
            tag = _split_tag(base_cfg, args.seed)
            if tag not in seen_tabular:
                ensure_entities(dataset, split_tag=tag, force=args.force)
                ensure_tabular_features(dataset, split_tag=tag, force=args.force)
                ensure_text(
                    dataset,
                    split_tag=tag,
                    dictionaries=CAMEO_DICTIONARIES,
                    force=args.force,
                )
                seen_tabular.add(tag)

        # --------------------------------------------------
        # Stage 6a: MLP experiments
        # --------------------------------------------------
        for base_cfg in mlp_configs:
            tag = _split_tag(base_cfg, args.seed)
            cfg = {**base_cfg, "dataset": dataset, "seed": args.seed, "split_tag": tag}
            cfg["exp_id"] = hash_config(cfg)
            logger.info(
                f"[MLP] Dataset={dataset} | Split={tag} | "
                f"Config={base_cfg['_config_path']} | ID={cfg['exp_id']}"
            )
            run_mlp(cfg)

        # --------------------------------------------------
        # Stage 6b: GNN experiments
        # --------------------------------------------------
        for base_cfg in gnn_configs:
            tag = _split_tag(base_cfg, args.seed)
            cfg = {**base_cfg, "dataset": dataset, "seed": args.seed, "split_tag": tag}
            cfg["exp_id"] = hash_config(cfg)
            logger.info(
                f"[GNN] Dataset={dataset} | Split={tag} | "
                f"Config={base_cfg['_config_path']} | ID={cfg['exp_id']}"
            )
            run_gnn(cfg)

        # --------------------------------------------------
        # Stage 6c: BERT experiments
        # --------------------------------------------------
        for base_cfg in bert_configs:
            tag = _split_tag(base_cfg, args.seed)
            cfg = {**base_cfg, "dataset": dataset, "seed": args.seed, "split_tag": tag}
            cfg["exp_id"] = hash_config(cfg)
            logger.info(
                f"[BERT] Dataset={dataset} | Split={tag} | "
                f"Config={base_cfg['_config_path']} | ID={cfg['exp_id']}"
            )
            run_bert(cfg)

        # --------------------------------------------------
        # Stage 6d: Multi-Domain Geometric experiments
        # --------------------------------------------------
        for base_cfg in multi_domain_configs:
            tag = _split_tag(base_cfg, args.seed)
            logger.info("-" * 60)
            logger.info(
                f"[MULTI-DOMAIN] Dataset={dataset} | "
                f"Split={tag} | "
                f"Config={base_cfg['_config_path']} | "
                f"Seed={args.seed}"
            )
            logger.info("-" * 60)
            run_multi_domain(
                cfg=base_cfg,
                dataset_name=dataset,
                split_tag=tag,
                seed=args.seed,
            )

    logger.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
