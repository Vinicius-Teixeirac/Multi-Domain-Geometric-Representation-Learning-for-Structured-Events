import json
from pathlib import Path
from typing import Optional, Tuple, Dict

from src.config.paths import ARTIFACTS_DATA, RESULTS_DIR


def _find_checkpoint(dataset: str, exp_id: str) -> Optional[Path]:
    """Return the best_model.pt path for exp_id under any model subdir, or None."""
    if not exp_id:
        return None
    base = ARTIFACTS_DATA / dataset / "models"
    if not base.exists():
        return None
    for model_dir in base.iterdir():
        ck = model_dir / exp_id / "best_model.pt"
        if ck.exists():
            return ck
    return None


def _find_results_json(dataset: str, exp_id: str) -> Optional[Path]:
    """Scan RESULTS_DIR for a JSON file whose exp_id field matches, or return None."""
    if not exp_id:
        return None
    base = RESULTS_DIR / dataset
    if not base.exists():
        return None
    for p in base.rglob("*.json"):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("exp_id") == exp_id:
            return p
    return None


def should_skip(exp_id: str, dataset: str) -> Tuple[bool, Dict[str, Optional[str]]]:
    """Decide whether an experiment should be skipped based on existing artifacts.

    Returns (skip, info) where info contains keys `checkpoint` and/or `results_file` when available.
    """
    info = {"checkpoint": None, "results_file": None}
    if not exp_id:
        return False, info

    ck = _find_checkpoint(dataset, exp_id)
    if ck is not None:
        info["checkpoint"] = str(ck)
        return True, info

    res = _find_results_json(dataset, exp_id)
    if res is not None:
        info["results_file"] = str(res)
        return True, info

    return False, info
