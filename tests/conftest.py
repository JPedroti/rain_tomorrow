"""Shared fixtures: a small synthetic frame with the weatherAUS schema, plus real-data / artifact fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config

DIRECTIONS = ["N", "NNE", "NE", "E", "SE", "S", "SW", "W", "NW"]


def make_weather_frame(n: int = 600, seed: int = 0, start: str = "2010-01-01", end: str = "2016-12-31",
                       locations=("Sydney", "Perth", "Darwin")) -> pd.DataFrame:
    """Random rows with the raw schema; rain depends on Humidity3pm so models have signal."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.integers(pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9, n), unit="s").normalize()
    df = pd.DataFrame({config.DATE_COL: dates, config.LOCATION_COL: rng.choice(locations, n)})
    for col in config.NUMERIC_FEATURES:
        df[col] = rng.normal(50, 15, n)
    df["Rainfall"] = rng.exponential(2.0, n) * (rng.random(n) < 0.4)
    df["Evaporation"] = rng.exponential(4.0, n)
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = rng.choice(DIRECTIONS, n)
    df["RainToday"] = np.where(df["Rainfall"] > 1.0, "Yes", "No")
    logit = (df["Humidity3pm"] - 55) / 8
    df[config.TARGET_COL] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    # sprinkle missing values, including a structurally missing column for one location
    for col in ["Sunshine", "Pressure9am", "WindDir9am", "Cloud3pm"]:
        df.loc[rng.random(n) < 0.15, col] = np.nan
    df.loc[df[config.LOCATION_COL] == locations[-1], "Evaporation"] = np.nan
    return df


@pytest.fixture
def weather_frame() -> pd.DataFrame:
    return make_weather_frame()


requires_data = pytest.mark.skipif(not config.DATA_PATH.exists(), reason="data/weatherAUS.csv not available")
requires_model = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.METADATA_PATH.exists()), reason="frozen model not trained yet")


@pytest.fixture(scope="session")
def dev_split():
    from src.data import load_train_test

    return load_train_test()
