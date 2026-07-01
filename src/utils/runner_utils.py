# src/utils/runner_utils.py
"""Shared helper utilities used by all experiment runner modules.

Covers GPU/hardware introspection for results provenance (collect_gpu_info),
YAML config loading, parameter counting, JSON-safe result serialization, and
timestamped results-file persistence -- the boilerplate every run_*() runner
needs around its actual train/evaluate calls.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

__all__ = [
    "collect_gpu_info",
    "count_trainable_parameters",
    "load_yaml_config",
    "make_json_serializable",
    "save_runner_results",
]


def collect_gpu_info(device: str) -> Dict[str, Any]:
    """
    Return GPU name, index, total memory, and isolation status for the given device.

    Parameters
    ----------
    device : str
        Target device string (e.g. 'cuda:0', 'cpu'). Non-"cuda"-prefixed
        devices short-circuit to an all-None dict.

    Returns
    -------
    dict with keys:
        gpu_name : str or None
        gpu_index : int or None
        gpu_total_memory_mb : int or None
        isolated : bool or None
            See _gpu_isolated for the tri-state semantics.
    All values fall back to None (rather than raising) if CUDA is
    unavailable or GPU introspection fails for any reason.
    """
    if not device.startswith("cuda"):
        return {
            "gpu_name": None,
            "gpu_index": None,
            "gpu_total_memory_mb": None,
            "isolated": None,
        }
    try:
        import torch
        gpu_index = int(device.split(":")[1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(gpu_index)
        return {
            "gpu_name": props.name,
            "gpu_index": gpu_index,
            "gpu_total_memory_mb": props.total_memory // (1024 * 1024),
            "isolated": _gpu_isolated(gpu_index),
        }
    except Exception:
        return {
            "gpu_name": None,
            "gpu_index": None,
            "gpu_total_memory_mb": None,
            "isolated": None,
        }


def _gpu_isolated(gpu_index: int) -> Optional[bool]:
    """
    Check whether this process has exclusive use of the given GPU.

    Parameters
    ----------
    gpu_index : int
        CUDA device index to query via nvidia-smi.

    Returns
    -------
    bool or None
        True if no other compute processes share this GPU, False if other
        PIDs are present, or None if isolation could not be determined
        (nvidia-smi missing/failed/timed out) -- a tri-state result since
        "undetermined" must not be conflated with "not isolated".
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
                "-i", str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=5,  # bound the call so a hung nvidia-smi can't stall the pipeline
        )
        if result.returncode != 0:
            return None
        pids = [
            int(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip().isdigit()
        ]
        return all(p == os.getpid() for p in pids)
    except Exception:
        return None


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Return the total number of initialized, trainable parameters (skips lazy params).

    Parameters
    ----------
    model : nn.Module
        Model to count parameters for.

    Returns
    -------
    int
        Sum of `numel()` over parameters with `requires_grad=True`,
        excluding any `UninitializedParameter` (lazy layers not yet built).
    """
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, UninitializedParameter):
            continue
        total += p.numel()
    return total


def save_runner_results(results: dict, results_dir: Path, prefix: str) -> Path:
    """
    Write results as a timestamped JSON file under results_dir.

    Parameters
    ----------
    results : dict
        JSON-serializable results dict (see make_json_serializable).
    results_dir : Path
        Directory to write into; created if missing.
    prefix : str
        Filename prefix (typically the model family, e.g. "mlp", "bert").

    Returns
    -------
    Path
        Path to the written `{prefix}_results_{timestamp}.json` file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = results_dir / f"{prefix}_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def make_json_serializable(obj: object) -> object:
    """
    Recursively convert numpy scalars/arrays to native Python types for json.dump.

    Parameters
    ----------
    obj : object
        Arbitrary nested structure of dicts/lists possibly containing numpy
        scalars or arrays (as produced by metrics computation).

    Returns
    -------
    object
        Same structure with numpy types replaced by plain Python
        int/float/list; other types are returned unchanged.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    return obj
