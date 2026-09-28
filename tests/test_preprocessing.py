"""Preprocessing: statistics learned on TRAIN only, no refit on transform, robust inputs, no leaking columns."""
import copy

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import make_weather_frame
from src import config
from src.data import encode_target
from src.modeling import build_pipeline
from src.preprocessing import DayOfYearEncoder, input_columns


def _fit(frame, **kwargs):
    params = {"feature_set": "full", "preprocessing": "indicators", **kwargs}
    return build_pipeline(**params).fit(frame, encode_target(frame[config.TARGET_COL]))


def _numeric_imputer(pipe):
    return pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]


def test_imputer_and_scaler_statistics_equal_train_statistics():
    train = make_weather_frame(n=500, seed=1)
    pipe = _fit(train)
    ct = pipe.named_steps["preprocess"]
    numeric_cols = next(t[2] for t in ct.transformers_ if t[0] == "num")
    np.testing.assert_allclose(_numeric_imputer(pipe).statistics_, train[numeric_cols].median().to_numpy())

    scaler = ct.named_transformers_["num"].named_steps["scale"]
    imputed = _numeric_imputer(pipe).transform(train[numeric_cols])
    np.testing.assert_allclose(scaler.mean_, imputed.mean(axis=0))

    onehot = ct.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = next(t[2] for t in ct.transformers_ if t[0] == "cat")
    for col, cats in zip(cat_cols, onehot.categories_):
        expected = set(train[col].dropna().astype(str)) | ({"missing"} if train[col].isna().any() else set())
        assert set(cats) == expected


def test_transform_and_predict_never_refit():
    train = make_weather_frame(n=500, seed=2)
    other = make_weather_frame(n=300, seed=3, start="2017-01-01", end="2017-06-30")
    other[config.NUMERIC_FEATURES] = other[config.NUMERIC_FEATURES] * 3 + 100  # very different distribution
    pipe = _fit(train)
    before = joblib.hash(pipe)
    pipe.predict_proba(other)
    pipe.named_steps["preprocess"].transform(other)
    assert joblib.hash(pipe) == before


def test_extra_columns_and_column_order_do_not_change_predictions():
    train = make_weather_frame(n=500, seed=4)
    pipe = _fit(train)
    new = make_weather_frame(n=80, seed=5)
    reference = pipe.predict_proba(new.drop(columns=[config.TARGET_COL]))[:, 1]
    polluted = new.assign(RISK_MM=99.0, junk="x")
    polluted = polluted[list(reversed(polluted.columns))]
    np.testing.assert_allclose(pipe.predict_proba(polluted)[:, 1], reference)


def test_model_features_exclude_target_future_and_year():
    pipe = _fit(make_weather_frame(n=300, seed=6))
    names = " ".join(pipe.named_steps["preprocess"].get_feature_names_out()).lower()
    for forbidden in ["raintomorrow", "risk_mm", "year"]:
        assert forbidden not in names
    assert "date__doy_sin" in names and "date__doy_cos" in names


def test_day_of_year_encoder_values_and_input_types():
    enc = DayOfYearEncoder().fit(None)
    as_text = pd.DataFrame({"Date": ["2015-01-01", "2015-07-02", "2015-12-31"]})
    out = enc.transform(as_text)
    np.testing.assert_allclose(out[0], [0.0, 1.0], atol=1e-12)
    one_day = 2 * np.pi / 365.25
    assert np.linalg.norm(out[2] - out[0]) < 2 * one_day  # 31-Dec is next to 1-Jan on the circle
    assert np.linalg.norm(out[1] - out[0]) > 1.9           # early July is on the opposite side
    as_datetime = as_text.assign(Date=pd.to_datetime(as_text["Date"]))
    np.testing.assert_allclose(enc.transform(as_datetime), out)
    with pytest.raises(ValueError):
        enc.transform(pd.DataFrame({"Date": ["2015-01-01", None]}))


def test_unknown_categories_and_all_missing_columns_are_handled():
    pipe = _fit(make_weather_frame(n=400, seed=7))
    new = make_weather_frame(n=20, seed=8)
    new["Location"] = "Atlantis"
    new["WindDir3pm"] = np.nan  # all-missing column parsed as float
    new["WindDir3pm"] = new["WindDir3pm"].astype(float)
    proba = pipe.predict_proba(new)[:, 1]
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()


def test_log_variant_transforms_only_skewed_features():
    pipe = _fit(make_weather_frame(n=300, seed=9), preprocessing="indicators_log")
    ct = pipe.named_steps["preprocess"]
    log_cols = next(t[2] for t in ct.transformers_ if t[0] == "num_log")
    assert log_cols == config.SKEWED_FEATURES


def test_feature_sets_are_consistent_with_measurement_windows():
    for name, spec in config.FEATURE_SETS.items():
        assert set(spec["numeric"]) <= set(config.NUMERIC_FEATURES)
        assert set(spec["categorical"]) <= set(config.CATEGORICAL_FEATURES)
        assert config.TARGET_COL not in input_columns(name)
    morning = config.FEATURE_SETS["morning"]
    for feature in morning["numeric"] + morning["categorical"]:
        window = config.FEATURE_WINDOWS[feature]
        assert "t+1" not in window and "15:00" not in window, feature
    assert "MaxTemp" not in config.FEATURE_SETS["full_without_maxtemp"]["numeric"]
    assert len(config.RAW_FEATURES) == 21


def test_final_feature_set_is_closed_at_the_prediction_moment():
    """Final model = forecast at the end of day t: no window may close after 00:00 of t+1."""
    spec = config.FEATURE_SETS[config.FINAL_FEATURE_SET]
    features = spec["numeric"] + spec["categorical"]
    assert len(features) == 20 and "MaxTemp" not in features
    for feature in features:
        assert "09:00 de t+1" not in config.FEATURE_WINDOWS[feature], feature


def test_pipeline_is_deterministic():
    frame = make_weather_frame(n=400, seed=10)
    a = _fit(frame).named_steps["model"].coef_
    b = _fit(copy.deepcopy(frame)).named_steps["model"].coef_
    np.testing.assert_array_equal(a, b)
