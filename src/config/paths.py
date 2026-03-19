# src/config/paths.py

# Importing this module guarantees all project directories exist.
# This file is intentionally side-effectful.

from pathlib import Path

# -----------------------------------------------------------------------------
# Project root
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------------------------------------------
# Data roots
# -----------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data"

RAW_DATA = DATA_ROOT / "raw"              # Original GDELT samples
PROCESSED_DATA = DATA_ROOT / "processed"  # Cleaned + selected columns
SPLITS_DATA = DATA_ROOT / "splits"        # Train/val/test splits
FEATURES_DATA = DATA_ROOT / "features"    # Numeric features / tensors
GRAPHS_DATA = DATA_ROOT / "graphs"        # Graph objects
TEXT_DATA = DATA_ROOT / "text"            # NLP sentences / tokenized inputs
ARTIFACTS_DATA = DATA_ROOT / "artifacts"  # Encoders, mappings, scalers
ENTITIES_DATA = DATA_ROOT / "entities"    # Node entities

# -----------------------------------------------------------------------------
# Logs & outputs
# -----------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# -----------------------------------------------------------------------------
# Create folders if missing
# -----------------------------------------------------------------------------
for p in [
    RAW_DATA,
    PROCESSED_DATA,
    SPLITS_DATA,
    FEATURES_DATA,
    GRAPHS_DATA,
    TEXT_DATA,
    ARTIFACTS_DATA,
    LOGS_DIR,
    RESULTS_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)
