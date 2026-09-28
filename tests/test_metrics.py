"""ROC-AUC, PR-AUC, KS, bootstrap, windowed metrics, threshold tables and the 1-SE selection rule."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

from src.metrics import (
    bootstrap_ci,
    discrimination_metrics,
    ks_statistic,
    metrics_by_period,
    threshold_metrics,
)
from src.modeling import select_one_se


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.22).astype(int)
    score = 1 / (1 + np.exp(-(rng.normal(0, 1, 4000) + 1.5 * y)))
    return y, score


def test_ks_equals_two_sample_ks_between_class_scores(scored):
    y, s = scored
    assert ks_statistic(y, s) == pytest.approx(ks_2samp(s[y == 1], s[y == 0]).statistic, abs=1e-12)


def test_metrics_match_sklearn_definitions(scored):
    y, s = scored
    m = discrimination_metrics(y, s)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s))
    assert m["prevalence"] == pytest.approx(y.mean())
    assert m["status"] == "ok"


def test_perfect_inverted_and_uninformative_scores():
    y = np.array([0] * 50 + [1] * 50)
    perfect = discrimination_metrics(y, np.linspace(0, 1, 100))
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0 and perfect["ks"] == 1.0
    inverted = discrimination_metrics(y, np.linspace(1, 0, 100))
    assert inverted["roc_auc"] == 0.0
    rng = np.random.default_rng(1)
    y_big = (rng.random(20000) < 0.2).astype(int)
    noise = discrimination_metrics(y_big, rng.random(20000))
    assert noise["roc_auc"] == pytest.approx(0.5, abs=0.02)
    assert noise["pr_auc"] == pytest.approx(y_big.mean(), abs=0.02)
    assert noise["ks"] < 0.05


def test_unusable_samples_return_nan_with_reason():
    single = discrimination_metrics(np.zeros(500), np.random.default_rng(0).random(500), min_rows=200, min_per_class=30)
    assert single["status"] == "single_class" and np.isnan(single["roc_auc"])
    small = discrimination_metrics(np.array([0, 1] * 50), np.linspace(0, 1, 100), min_rows=200, min_per_class=30)
    assert small["status"] == "insufficient_volume" and np.isnan(small["ks"])


def test_bootstrap_ci_is_deterministic_and_brackets_estimate(scored):
    y, s = scored
    ci = bootstrap_ci(y, s, n_boot=200, seed=3)
    assert ci == bootstrap_ci(y, s, n_boot=200, seed=3)
    point = discrimination_metrics(y, s)
    for metric, (low, high) in ci.items():
        assert low < point[metric] < high


def test_metrics_by_period_splits_by_month_and_flags_small_windows(scored):
    y, s = scored
    dates = pd.Series(pd.to_datetime(["2017-01-15"] * 3900 + ["2017-02-10"] * 100))
    table = metrics_by_period(dates, y, s, with_ci=False)
    assert table["period"].tolist() == ["2017-01", "2017-02"]
    assert table.loc[0, "status"] == "ok" and table.loc[0, "n"] == 3900
    assert table.loc[1, "status"] == "insufficient_volume" and np.isnan(table.loc[1, "roc_auc"])
    jan = discrimination_metrics(y[:3900], s[:3900])
    assert table.loc[0, "roc_auc"] == pytest.approx(jan["roc_auc"])


def test_threshold_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.6, 0.2, 0.7, 0.4, 0.3, 0.1, 0.05])
    m = threshold_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 4, 1)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["f1"] == pytest.approx(2 / 3)
    lower = threshold_metrics(y, s, 0.15)
    assert lower["recall"] >= m["recall"] and lower["specificity"] <= m["specificity"]


def test_one_standard_error_rule_prefers_first_candidate_within_band():
    results = pd.DataFrame({
        "order": [0, 1, 2, 3],
        "roc_auc_mean": [0.870, 0.8795, 0.880, 0.860],
        "roc_auc_se": [0.001, 0.001, 0.001, 0.001],
    })
    assert select_one_se(results) == 1  # 0.8795 >= 0.880 - 0.001; candidate 0 is outside the band
    results.loc[0, "roc_auc_mean"] = 0.8792
    assert select_one_se(results) == 0
