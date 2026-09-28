"""Audit of the frozen artifacts: hash, selection protocol, TRAIN-only statistics, OOT isolation."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import requires_data, requires_model
from src import config
from src.artifacts import load_frozen_model
from src.data import file_sha256
from src.modeling import select_one_se

pytestmark = [requires_data, requires_model]


@pytest.fixture(scope="module")
def frozen():
    model, metadata = load_frozen_model()
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    return model, metadata, metrics


def test_metadata_records_hash_cutoff_and_features(frozen):
    _, metadata, _ = frozen
    assert metadata["model_sha256"] == file_sha256(config.MODEL_PATH)
    tp = metadata["temporal_protocol"]
    assert tp["oot_start"] == config.OOT_START
    assert tp["train_date_range"][1] < config.OOT_START
    assert tp["oot_read_by_training"] is False
    assert metadata["features"]["feature_set"] == config.FINAL_FEATURE_SET
    assert config.TARGET_COL not in metadata["features"]["input_columns"]


def test_metrics_json_has_required_metrics(frozen):
    _, _, metrics = frozen
    for key in ("roc_auc", "pr_auc", "ks"):
        assert 0.0 <= metrics["test"][key] <= 1.0
        if metrics.get("oot"):
            assert 0.0 <= metrics["oot"][key] <= 1.0


def test_frozen_params_follow_the_preregistered_rule(frozen):
    _, metadata, metrics = frozen
    cv = pd.read_csv(config.CV_RESULTS_PATH)
    stage1 = cv[cv["stage"] == "1_preprocessing"].reset_index(drop=True)
    stage2 = cv[cv["stage"] == "2_hyperparameters"].reset_index(drop=True)
    assert stage1.loc[select_one_se(stage1), "preprocessing"] == metadata["model"]["preprocessing_variant"]
    chosen = stage2.loc[select_one_se(stage2)]
    assert chosen["C"] == metadata["model"]["C"]
    assert chosen["l1_ratio"] == metadata["model"]["l1_ratio"]
    cw = None if pd.isna(chosen["class_weight"]) else chosen["class_weight"]
    assert cw == metadata["model"]["class_weight"]
    assert metrics["selected_params"]["C"] == metadata["model"]["C"]


def test_model_statistics_were_learned_on_train_only(frozen, dev_split):
    """Imputer medians, scaler means and categories of the frozen pipeline are TRAIN statistics.

    Medians of 0.1-rounded measurements coincide for TRAIN and TRAIN+TEST, so the discriminating check
    is the scaler: its means must equal the TRAIN means and differ from the TRAIN+TEST means.
    """
    model, _, _ = frozen
    train, test = dev_split["train"], dev_split["test"]
    ct = model.named_steps["preprocess"]
    for name, transformer, columns in ct.transformers_:
        if name.startswith("num"):
            imputer, scaler = transformer.named_steps["impute"], transformer.named_steps["scale"]
            prep = (lambda f: np.log1p(f)) if name == "num_log" else (lambda f: f)
            np.testing.assert_allclose(imputer.statistics_, prep(train[columns]).median().to_numpy())
            train_means = imputer.transform(prep(train[columns])).mean(axis=0)
            both_means = imputer.transform(prep(pd.concat([train, test])[columns])).mean(axis=0)
            np.testing.assert_allclose(scaler.mean_, train_means, rtol=0, atol=1e-9)
            assert not np.allclose(scaler.mean_, both_means, rtol=0, atol=1e-9)
        if name == "cat":
            for col, cats in zip(columns, transformer.named_steps["onehot"].categories_):
                expected = set(train[col].dropna().astype(str)) | ({"missing"} if train[col].isna().any() else set())
                assert set(cats) == expected
