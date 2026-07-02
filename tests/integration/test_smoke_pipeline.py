"""Full-pipeline smoke test: every model family, trained end-to-end, in one process.

Opt-in integration test (see the `integration` marker in pyproject.toml).
Not part of the default `pytest` run: it needs a GPU, takes several
minutes, and writes real artifacts under data/ and results/. Run it
explicitly with `pytest -m integration`.

Runs all 13 smoke configs through a single `main.py` invocation on purpose:
this exact shape (many models trained back to back in one process)
previously surfaced a CUDA OOM from GPU memory not being released between
sequential runs, and an idempotency bug where a crashed run's stray
checkpoint made retries silently skip with no results ever produced. A
test that spawned one subprocess per model would never exercise that path.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "sample_500"
SMOKE_CONFIG_DIR = REPO_ROOT / "src" / "config" / "model_setup" / "smoke"

EXPECTED_RESULT_COUNTS = {
    "EventMLP": 1,
    "HomogeneousGNN": 3,       # graphSAGE, gat, gin
    "HeterogeneousGNN": 3,     # han, rgat, rgcn
    "BertForQuadClass": 1,
    "MultiDomainGeometricModel": 5,
}


def _clear_dataset_artifacts() -> None:
    """Remove derived artifacts for DATASET so the run is a real rebuild, not an idempotent skip."""
    for root_name in ("processed", "splits", "entities", "features", "text", "artifacts"):
        shutil.rmtree(REPO_ROOT / "data" / root_name / DATASET, ignore_errors=True)
    shutil.rmtree(REPO_ROOT / "results" / DATASET, ignore_errors=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="smoke configs are pinned to training.device: cuda",
)
def test_all_models_train_and_evaluate_in_one_process():
    """Run every model family's smoke config through one main.py invocation."""
    _clear_dataset_artifacts()

    cmd = [
        sys.executable, "main.py",
        "--datasets", DATASET,
        "--mlp-configs", str(SMOKE_CONFIG_DIR / "mlp.yaml"),
        "--gnn-configs",
        str(SMOKE_CONFIG_DIR / "graphSAGE.yaml"),
        str(SMOKE_CONFIG_DIR / "gat.yaml"),
        str(SMOKE_CONFIG_DIR / "gin.yaml"),
        str(SMOKE_CONFIG_DIR / "han.yaml"),
        str(SMOKE_CONFIG_DIR / "rgat.yaml"),
        str(SMOKE_CONFIG_DIR / "rgcn.yaml"),
        "--bert-configs", str(SMOKE_CONFIG_DIR / "bert.yaml"),
        "--multi-domain-configs",
        str(SMOKE_CONFIG_DIR / "multi_domain.yaml"),
        str(SMOKE_CONFIG_DIR / "multi_domain_riemannian.yaml"),
        str(SMOKE_CONFIG_DIR / "multi_domain_riemannian_aware.yaml"),
        str(SMOKE_CONFIG_DIR / "multi_domain_gat_attention.yaml"),
        str(SMOKE_CONFIG_DIR / "multi_domain_region_fourier_gated.yaml"),
    ]

    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    combined_output = proc.stdout + proc.stderr

    assert proc.returncode == 0, (
        f"main.py exited with {proc.returncode}\n--- tail of output ---\n{combined_output[-4000:]}"
    )
    assert "All experiments completed successfully." in combined_output

    results_dir = REPO_ROOT / "results" / DATASET
    for model_family, expected_count in EXPECTED_RESULT_COUNTS.items():
        result_files = sorted((results_dir / model_family).glob("*.json"))
        assert len(result_files) == expected_count, (
            f"{model_family}: expected {expected_count} result file(s), found {len(result_files)}"
        )
        for path in result_files:
            metrics = json.loads(path.read_text())["metrics"]
            assert metrics["accuracy"] == metrics["accuracy"]  # not NaN
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert metrics["f1_macro"] == metrics["f1_macro"]  # not NaN
