"""Feature preprocessing: one ColumnTransformer shared by training, evaluation and inference.

    numeric      -> [log1p on skewed columns] -> median imputer [+ missing indicators] -> StandardScaler
    categorical  -> cast to object -> constant "missing" imputer -> OneHotEncoder(handle_unknown="ignore")
    Date         -> day-of-year sin/cos (no year: 2017 would be outside the training support)

Every statistic (medians, means, scales, categories) is learned in `fit`, which only ever sees TRAIN.
Columns that are not listed (e.g. RainTomorrow, extra columns in new files) are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

MISSING_CATEGORY = "missing"


class DayOfYearEncoder(TransformerMixin, BaseEstimator):
    """Stateless encoder: Date -> (sin, cos) of the day of the year.

    The cyclical form keeps 31-Dec and 1-Jan close and needs nothing learned from data.
    """

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        col = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).ravel())
        dates = pd.to_datetime(col, format="%Y-%m-%d") if not pd.api.types.is_datetime64_any_dtype(col) else col
        if dates.isna().any():
            raise ValueError("Date contains missing or unparseable values.")
        angle = 2.0 * np.pi * (dates.dt.dayofyear.to_numpy() - 1) / 365.25
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features=None):
        return np.array(["doy_sin", "doy_cos"], dtype=object)


def to_object(X):
    """Cast categorical columns to object with np.nan as the only missing marker.

    Makes the pipeline indifferent to how a CSV was parsed (pandas `str` dtype, an all-NaN float column
    in a new file, ...).
    """
    X = pd.DataFrame(X).astype(object)
    return X.where(X.notna(), np.nan)


def _numeric_branch(missing_indicators: bool, log_transform: bool) -> Pipeline:
    steps = []
    if log_transform:
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps += [
        ("impute", SimpleImputer(strategy="median", add_indicator=missing_indicators)),
        ("scale", StandardScaler()),
    ]
    return Pipeline(steps)


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    missing_indicators: bool,
    log_transform: bool,
) -> ColumnTransformer:
    skewed = [c for c in numeric if log_transform and c in config.SKEWED_FEATURES]
    regular = [c for c in numeric if c not in skewed]

    transformers = [("num", _numeric_branch(missing_indicators, log_transform=False), regular)]
    if skewed:
        transformers.append(("num_log", _numeric_branch(missing_indicators, log_transform=True), skewed))
    transformers += [
        (
            "cat",
            Pipeline([
                ("to_object", FunctionTransformer(to_object, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
        ("date", DayOfYearEncoder(), [config.DATE_COL]),
    ]
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def input_columns(feature_set: str) -> list[str]:
    """Raw columns the pipeline for `feature_set` needs in its input frame."""
    spec = config.FEATURE_SETS[feature_set]
    return [config.DATE_COL, *spec["numeric"], *spec["categorical"]]
