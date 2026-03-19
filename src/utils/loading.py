# src/utils/loading.py
import json
from pathlib import Path
from typing import Any, Dict
import warnings

import pandas as pd


def load_json(path: Path | str) -> Dict[str, Any]:
    """
    Read a UTF-8 JSON file and return its contents.

    Centralized to avoid repeated boilerplate and to
    standardize encoding / error handling.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_split(
    data_dir: Path,
    dataset_name: str,
    split: str,
    split_tag: str = "default",
) -> pd.DataFrame:
    """
    Load a preprocessed dataset split.

    Expected path:
        {data_dir}/{dataset_name}/{split}_{split_tag}.parquet
    """
    split_dir = data_dir / dataset_name
    filename = f"{split}_{split_tag}.parquet"

    return load_parquet(filename, split_dir)

def load_parquet(
    filename: str,
    directory: Path,
) -> pd.DataFrame:
    """
    Load a parquet file and perform minimal structural validation.

    Parameters
    ----------
    filename : str
        Parquet filename (with or without `.parquet` extension).
    directory : Path
        Directory containing the parquet file.
    allowed_columns : Optional[Iterable[str]]
        If provided, all columns in the file must be a subset of this list.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the dataframe is empty or violates column constraints.
    """

    # Normalize filename
    if not filename.endswith(".parquet"):
        filename = f"{filename}.parquet"

    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Loaded parquet file is empty: {path}")

    # Warn if index is not default (common silent bug)
    if not isinstance(df.index, pd.RangeIndex):
        warnings.warn(
            "Loaded DataFrame has a non-default index. "
            "Consider resetting index explicitly.",
            UserWarning,
        )

    return df
