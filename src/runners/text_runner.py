# src/runners/text_runner.py
"""
Runner for the natural-language event-text pipeline stage.

Wraps build_event_texts with a per-split idempotency check: unlike the
other runners, which skip the whole dataset when all expected outputs
exist, this runner loops over the train/valid/test splits individually
and skips (or builds) each one independently. Text produced here is
consumed by the BERT runner.
"""

import pandas as pd

from typing import Any, Dict

from src.utils.loading import load_parquet
from src.config.paths import SPLITS_DATA, TEXT_DATA
from src.representation.text.text_builder import build_event_texts


def ensure_text(
    dataset: str,
    split_tag: str,
    dictionaries: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Build natural-language event text (CAMEO-resolved) and cache it.

    Loops over the train/valid/test splits independently: for each split,
    if the cached text parquet already exists and ``force`` is False, that
    split is skipped; if the corresponding input split parquet is missing
    (e.g. no validation split for this sample), that split is silently
    skipped as well. This differs from the other runners' whole-dataset
    skip check, since text can legitimately be built incrementally per
    split.

    Parameters
    ----------
    dataset : str
        Dataset name (stem of the parquet file in data/raw/).
    split_tag : str
        Split identifier whose train/valid/test text files are checked
        and (re)built.
    dictionaries : dict
        CAMEO code-to-description lookup tables passed through to
        build_event_texts (e.g. {"EventCode": {...}}).
    force : bool
        If True, rebuild the text cache for a split even if it already
        exists. If False (default), an existing per-split cache is
        reused.

    Returns
    -------
    dict
        ``{"skipped": False, "dataset": str, "split_tag": str}``. Note
        this is always reported as not-skipped at the top level even
        when every individual split was skipped internally, since the
        skip decision here is per-split rather than whole-dataset.
    """

    in_dir = SPLITS_DATA / dataset
    out_dir = TEXT_DATA / dataset 
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        out_path = out_dir / f"{split}_{split_tag}_text.parquet"
        if out_path.exists() and not force:
            continue

        in_path = in_dir / f"{split}_{split_tag}.parquet"
        if not in_path.exists():
            continue

        df = load_parquet(f"{split}_{split_tag}.parquet", in_dir)

        df = df.copy()
        df["text"] = build_event_texts(df, dictionaries)
        # Rename to "label" for alignment with HF/text-pipeline label-column
        # conventions expected by the BERT datamodule/tokenizer downstream.
        df = df.rename(columns={"QuadClass": "label"})

        df[["text", "label"]].to_parquet(out_path)
    return {"skipped": False, "dataset": dataset, "split_tag": split_tag}
