"""Data loading and the temporal / random splits.

The OOT boundary is enforced here: development code paths call `load_development_data`, which drops
every row dated on/after `OOT_START` right after parsing, and `assert_no_oot` guards every fit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

REQUIRED_COLUMNS = [config.DATE_COL, config.TARGET_COL, *config.RAW_FEATURES]


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Read weatherAUS.csv, validate its schema and parse Date. No row is dropped here."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {path}: {missing}")
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], format="%Y-%m-%d")
    return df


def split_development_oot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by calendar: development = Date < OOT_START, OOT = Date >= OOT_START."""
    cutoff = pd.Timestamp(config.OOT_START)
    is_oot = df[config.DATE_COL] >= cutoff
    dev, oot = df.loc[~is_oot].copy(), df.loc[is_oot].copy()
    assert len(dev) + len(oot) == len(df)
    return dev, oot


def assert_no_oot(df: pd.DataFrame) -> None:
    """Raise if any row belongs to the OOT period. Called before every fit."""
    if (df[config.DATE_COL] >= pd.Timestamp(config.OOT_START)).any():
        raise ValueError(f"OOT rows (Date >= {config.OOT_START}) reached a development code path.")


def load_development_data(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Development rows only (Date < OOT_START). The OOT rows are discarded immediately."""
    dev, _ = split_development_oot(load_raw(path))
    assert_no_oot(dev)
    return dev


def load_oot_data(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """OOT rows only. To be used exclusively after the model is frozen (scripts/evaluate.py)."""
    _, oot = split_development_oot(load_raw(path))
    return oot


def drop_missing_target(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows without RainTomorrow cannot be used to fit or to score; they are removed and counted."""
    keep = df[config.TARGET_COL].notna()
    return df.loc[keep].copy(), int((~keep).sum())


def encode_target(target: pd.Series) -> np.ndarray:
    """RainTomorrow -> 1 for "Yes" (positive class), 0 for "No". Any other value is an error."""
    allowed = {config.POSITIVE_LABEL, config.NEGATIVE_LABEL}
    unexpected = set(target.dropna().unique()) - allowed
    if unexpected or target.isna().any():
        raise ValueError(f"Target must be {sorted(allowed)} without NaN; found extra values {unexpected}.")
    return (target == config.POSITIVE_LABEL).astype(int).to_numpy()


def split_train_test(dev_labelled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random stratified 80/20 split of the labelled development rows (random_state fixed)."""
    assert_no_oot(dev_labelled)
    y = encode_target(dev_labelled[config.TARGET_COL])
    train, test = train_test_split(
        dev_labelled,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return train.copy(), test.copy()


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Features frame (every column but the target; the pipeline selects what it uses) and 0/1 target."""
    return df.drop(columns=[config.TARGET_COL]), encode_target(df[config.TARGET_COL])


def load_train_test(path: Path | str = config.DATA_PATH) -> dict:
    """Full development preparation: dev rows -> drop missing target -> stratified train/test split."""
    dev = load_development_data(path)
    labelled, n_dropped = drop_missing_target(dev)
    train, test = split_train_test(labelled)
    return {
        "dev": dev,
        "train": train,
        "test": test,
        "n_dev_rows": len(dev),
        "n_dropped_missing_target": n_dropped,
    }
