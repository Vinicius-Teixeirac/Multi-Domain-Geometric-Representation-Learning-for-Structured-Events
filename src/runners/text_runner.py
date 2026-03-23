# src/runners/text_runner.py
import pandas as pd

from src.utils.loading import load_parquet
from src.config.paths import SPLITS_DATA, TEXT_DATA
from src.representation.text.text_builder import build_event_texts


def ensure_text(
    dataset: str,
    split_tag: str,
    dictionaries: dict,
    force: bool = False,
):
    """
    Build natural-language event text (CAMEO-resolved) and cache it.
    """

    in_dir = SPLITS_DATA / dataset
    out_dir = TEXT_DATA / dataset 
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        out_path = out_dir / f"{split}_{split_tag}_text.parquet"
        if out_path.exists() and not force:
            continue

        df = load_parquet(f"{split}_{split_tag}.parquet", in_dir)

        df = df.copy()
        df["text"] = build_event_texts(df, dictionaries)
        df = df.rename(columns={"QuadClass": "label"})

        df[["text", "label"]].to_parquet(out_path)
    return {"skipped": False, "dataset": dataset, "split_tag": split_tag}
