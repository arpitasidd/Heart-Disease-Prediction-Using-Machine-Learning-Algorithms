from __future__ import annotations
import os
import io
import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
OPENML_NAME = "heart-disease"

UCI_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak",
    "slope","ca","thal","num"
]

def _try_load_local(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

def _try_load_uci() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(UCI_URL, header=None, names=UCI_COLUMNS)
        return df
    except Exception:
        return None

def _try_load_openml() -> pd.DataFrame | None:
    try:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name=OPENML_NAME, version=1, as_frame=True)
        df = ds.frame.copy()
        return df
    except Exception:
        return None

def _synthetic_sample(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age": rng.integers(29, 77, size=n),
        "sex": rng.integers(0, 2, size=n),
        "cp": rng.integers(0, 4, size=n),
        "trestbps": rng.integers(94, 200, size=n),
        "chol": rng.integers(126, 564, size=n),
        "fbs": rng.integers(0, 2, size=n),
        "restecg": rng.integers(0, 2, size=n),
        "thalach": rng.integers(71, 202, size=n),
        "exang": rng.integers(0, 2, size=n),
        "oldpeak": rng.normal(1.0, 1.16, size=n).clip(0, 6),
        "slope": rng.integers(0, 3, size=n),
        "ca": rng.integers(0, 4, size=n),
        "thal": rng.integers(0, 3, size=n),
        "num": rng.integers(0, 2, size=n),  # 0 = no disease, 1 = disease
    })
    return df

def load_heart_dataframe(data_dir: str = "data") -> pd.DataFrame:
    """Load heart disease dataset with fallbacks.
    Prepares a binary target column 'target' (1 = disease present).
    Returns a clean pandas DataFrame.
    """
    # 1) Local
    local_path = os.path.join(data_dir, "heart.csv")
    df = _try_load_local(local_path)
    source = "local"
    # 2) UCI
    if df is None:
        df = _try_load_uci()
        source = "uci"
    # 3) OpenML
    if df is None:
        df = _try_load_openml()
        source = "openml"
    # 4) Synthetic
    if df is None:
        df = _synthetic_sample()
        source = "synthetic"

    # Normalize schema -> ensure 'target' exists and is binary
    if "num" in df.columns and "target" not in df.columns:
        # UCI style: 'num' is 0..4; convert to binary
        df["target"] = (df["num"].astype(float) > 0).astype(int)
    elif "target" in df.columns:
        df["target"] = df["target"].astype(int)
    else:
        # If OpenML uses 'class' or other naming, standardize
        if "class" in df.columns:
            # Often 'present'/'absent' or 0/1
            df["target"] = df["class"].replace({"present":1, "absent":0}).astype(int)
        else:
            # As a last resort, create a target if missing (shouldn't happen)
            if "num" not in df.columns:
                df["target"] = 0

    # Replace '?' with NaN if present
    df = df.replace("?", pd.NA)

    # Coerce numeric columns where sensible
    for col in df.columns:
        if col == "target":
            continue
        # try numeric
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # Drop rows missing target
    df = df.dropna(subset=["target"])

    return df
