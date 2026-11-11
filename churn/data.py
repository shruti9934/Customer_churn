"""Loading and cleaning the IBM Telco Customer Churn dataset."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import DATA_PATH, DATA_URL, ID_COL, RAW_FEATURES, TARGET


def download_data(path: Path = DATA_PATH) -> Path:
    """Download the dataset if it is not already on disk."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and obvious data-quality issues.

    * ``TotalCharges`` is stored as text and is blank for brand-new customers
      (tenure 0) - convert to float and fill those with 0.
    * Map the ``Churn`` label Yes/No -> 1/0.
    * Drop exact duplicate rows.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True))
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].str.strip().map({"Yes": 1, "No": 0}).astype(int)
    df = df.drop_duplicates()
    return df


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return cleaned features ``X`` (raw columns) and target ``y``."""
    download_data(path)
    df = clean(pd.read_csv(path))
    X = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])[RAW_FEATURES]
    y = df[TARGET]
    return X, y
