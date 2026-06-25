import math
import pytest
import torch
import pandas as pd
import numpy as np


# ── Shared constants ──────────────────────────────────────────────

CATEGORICAL_CARDINALITIES = {"col_a": 10, "col_b": 50}
NUMERIC_DIM = 4
NUM_CLASSES = 4
BATCH_SIZE = 8


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def cardinalities():
    return dict(CATEGORICAL_CARDINALITIES)


@pytest.fixture
def sample_x_cat():
    return {
        name: torch.randint(0, card, (BATCH_SIZE,))
        for name, card in CATEGORICAL_CARDINALITIES.items()
    }


@pytest.fixture
def sample_x_num():
    return torch.randn(BATCH_SIZE, NUMERIC_DIM)


@pytest.fixture
def sample_dataframe():
    rng = np.random.default_rng(0)
    n = 32
    df = pd.DataFrame({
        "col_a": rng.integers(0, 10, size=n),
        "col_b": rng.integers(0, 50, size=n),
        "num_0": rng.standard_normal(n).astype(np.float32),
        "num_1": rng.standard_normal(n).astype(np.float32),
        "num_2": rng.standard_normal(n).astype(np.float32),
        "num_3": rng.standard_normal(n).astype(np.float32),
        "QuadClass": rng.integers(0, NUM_CLASSES, size=n),
    })
    return df


@pytest.fixture
def sample_dataframe_no_numeric():
    rng = np.random.default_rng(0)
    n = 16
    df = pd.DataFrame({
        "col_a": rng.integers(0, 10, size=n),
        "col_b": rng.integers(0, 50, size=n),
        "QuadClass": rng.integers(0, NUM_CLASSES, size=n),
    })
    return df


def expected_emb_dim(cardinality: int, rule: str = "sqrt") -> int:
    """Mirror of TabularInputEncoder._embedding_dim for test assertions."""
    if rule == "sqrt":
        return max(1, min(128, int(math.sqrt(cardinality))))
    if rule == "log":
        return max(1, min(128, int(math.log2(cardinality)) + 1))
    raise ValueError(rule)


def expected_output_dim(cardinalities: dict, numeric_dim: int, rule: str = "sqrt") -> int:
    """Expected total encoder output dim: sum of embedding dims + numeric dim."""
    return numeric_dim + sum(
        expected_emb_dim(c, rule) for c in cardinalities.values()
    )
