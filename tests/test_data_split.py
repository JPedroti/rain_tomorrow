"""Temporal separation (OOT) and the development TRAIN/TEST split."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_weather_frame, requires_data
from src import config
from src.data import (
    assert_no_oot,
    drop_missing_target,
    encode_target,
    load_development_data,
    load_raw,
    split_development_oot,
    split_train_test,
)

CUTOFF = pd.Timestamp(config.OOT_START)


def test_split_development_oot_on_synthetic_boundary():
    df = make_weather_frame(n=50)
    edge = pd.DataFrame({config.DATE_COL: pd.to_datetime(["2016-12-31", "2017-01-01"])})
    df = pd.concat([df, edge.assign(**{c: df[c].iloc[0] for c in df.columns if c != config.DATE_COL})], ignore_index=True)
    dev, oot = split_development_oot(df)
    assert dev[config.DATE_COL].max() == pd.Timestamp("2016-12-31")
    assert oot[config.DATE_COL].min() == CUTOFF
    assert len(dev) + len(oot) == len(df)


def test_assert_no_oot_guards_every_development_entry_point():
    df = make_weather_frame(n=100, start="2016-06-01", end="2017-03-01")
    df = df.dropna(subset=[config.TARGET_COL])
    with pytest.raises(ValueError, match="OOT"):
        assert_no_oot(df)
    with pytest.raises(ValueError, match="OOT"):
        split_train_test(df)
    from src.modeling import evaluate_candidates, run_model_selection

    y = encode_target(df[config.TARGET_COL])
    with pytest.raises(ValueError, match="OOT"):
        evaluate_candidates([{"feature_set": "full"}], df, y)
    with pytest.raises(ValueError, match="OOT"):
        run_model_selection(df, y, feature_set="full")


def test_encode_target_maps_positive_class_and_rejects_unknown():
    assert encode_target(pd.Series(["Yes", "No", "Yes"])).tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", "Maybe"]))
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", np.nan]))


@requires_data
def test_real_oot_is_exactly_2017_and_disjoint_from_development():
    raw = load_raw()
    dev, oot = split_development_oot(raw)
    assert dev[config.DATE_COL].max() < CUTOFF
    assert (oot[config.DATE_COL].dt.year == 2017).all()
    assert len(oot) == int((raw[config.DATE_COL].dt.year == 2017).sum())
    assert len(dev) + len(oot) == len(raw)
    assert not set(dev.index) & set(oot.index)


@requires_data
def test_development_loader_never_returns_oot_rows():
    dev = load_development_data()
    assert dev[config.DATE_COL].max() < CUTOFF


@requires_data
def test_train_test_are_disjoint_complete_and_free_of_oot(dev_split):
    train, test, dev = dev_split["train"], dev_split["test"], dev_split["dev"]
    labelled, n_dropped = drop_missing_target(dev)
    assert not set(train.index) & set(test.index)
    assert set(train.index) | set(test.index) == set(labelled.index)
    keys = ["Location", "Date"]
    assert train[keys].merge(test[keys], on=keys).empty
    for part in (train, test):
        assert part[config.DATE_COL].max() < CUTOFF
        assert part[config.TARGET_COL].notna().all()
    assert n_dropped == dev_split["n_dropped_missing_target"] == int(dev[config.TARGET_COL].isna().sum())


@requires_data
def test_split_is_80_20_stratified_and_reproducible(dev_split):
    from src.data import load_train_test

    train, test = dev_split["train"], dev_split["test"]
    n = len(train) + len(test)
    assert len(test) == int(np.ceil(config.TEST_SIZE * n))
    p_train = (train[config.TARGET_COL] == "Yes").mean()
    p_test = (test[config.TARGET_COL] == "Yes").mean()
    assert abs(p_train - p_test) < 1e-3
    again = load_train_test()
    assert again["train"].index.equals(train.index)
    assert again["test"].index.equals(test.index)
