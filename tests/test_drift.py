"""PSI (reference-learned bins, missing bin, empty bins) and KS drift."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from src.drift import ks_drift, positive_rate, psi_categorical, psi_numeric, quantile_edges


def test_psi_of_identical_distributions_is_zero():
    x = pd.Series(np.random.default_rng(0).normal(size=5000))
    value, _ = psi_numeric(x, x.copy())
    assert value == pytest.approx(0.0, abs=1e-12)


def test_psi_hand_computed_two_bins():
    reference = pd.Series([0.0] * 50 + [1.0] * 50)
    current = pd.Series([0.0] * 80 + [1.0] * 20)
    value, table = psi_numeric(reference, current, n_bins=2)
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert value == pytest.approx(expected, rel=1e-9)
    assert table["ref_pct"].tolist()[:2] == [0.5, 0.5]


def test_bins_come_from_reference_only():
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(0, 1, 3000))
    edges = quantile_edges(reference)
    for shift in (0.0, 5.0):
        _, table = psi_numeric(reference, pd.Series(rng.normal(shift, 1, 3000)))
        assert len(table) == len(edges) + 2  # bins + missing
        np.testing.assert_allclose(quantile_edges(reference), edges)
    big, _ = psi_numeric(reference, pd.Series(rng.normal(2.0, 1, 3000)))
    small, _ = psi_numeric(reference, pd.Series(rng.normal(0.05, 1, 3000)))
    assert big > 0.25 > small


def test_missing_values_are_their_own_bin_and_empty_bins_stay_finite():
    rng = np.random.default_rng(2)
    reference = pd.Series(rng.normal(size=2000))
    current = reference.copy()
    current.iloc[:1000] = np.nan
    value, table = psi_numeric(reference, current)
    assert np.isfinite(value) and value > 0.25
    assert table.loc[table["bin"] == "missing", "cur_pct"].item() == pytest.approx(0.5)
    far, _ = psi_numeric(reference, pd.Series(rng.normal(50, 1, 2000)))  # every current value in one bin
    assert np.isfinite(far)


def test_tied_zero_inflated_feature_gets_unique_edges():
    reference = pd.Series([0.0] * 700 + list(np.linspace(0.2, 30, 300)))
    edges = quantile_edges(reference)
    assert len(edges) == len(np.unique(edges)) and edges[0] == 0.0
    value, _ = psi_numeric(reference, reference.sample(frac=1.0, random_state=0))
    assert value == pytest.approx(0.0, abs=1e-12)


def test_ks_drift_matches_scipy_and_ignores_missing():
    rng = np.random.default_rng(3)
    a, b = pd.Series(rng.normal(0, 1, 1000)), pd.Series(rng.normal(0.3, 1, 800))
    b.iloc[:100] = np.nan
    stat, p = ks_drift(a, b)
    expected = ks_2samp(a, b.dropna())
    assert stat == pytest.approx(expected.statistic) and p == pytest.approx(expected.pvalue)


def test_categorical_psi_handles_unseen_and_missing():
    reference = pd.Series(["A"] * 50 + ["B"] * 50)
    same, _ = psi_categorical(reference, reference.copy())
    assert same == pytest.approx(0.0, abs=1e-12)
    changed, table = psi_categorical(reference, pd.Series(["A"] * 50 + ["C"] * 30 + [np.nan] * 20))
    assert changed > 0.25
    assert "unseen" in table["bin"].tolist()


def test_positive_rate_ignores_missing_target():
    assert positive_rate(pd.Series(["Yes", "No", np.nan, "No"])) == pytest.approx(1 / 3)
