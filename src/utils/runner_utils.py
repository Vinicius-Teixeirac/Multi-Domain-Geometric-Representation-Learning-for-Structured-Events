# src/utils/runner_utils.py
"""Shared helper utilities used by all experiment runner modules."""

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
    """Return GPU name, index, total memory, and isolation status for the given device."""
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
    """True if no other compute processes share this GPU; None if undetermined."""
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
            timeout=5,
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
    """Return the total number of initialized, trainable parameters (skips lazy params)."""
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, UninitializedParameter):
            continue
        total += p.numel()
    return total


def save_runner_results(results: dict, results_dir: Path, prefix: str) -> Path:
    """Write results as a timestamped JSON file under results_dir; return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = results_dir / f"{prefix}_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def make_json_serializable(obj: object) -> object:
    """Recursively convert numpy scalars/arrays to native Python types for json.dump."""
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
