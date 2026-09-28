"""Distribution-shift diagnostics: PSI (bins learned on the reference = TRAIN) and two-sample KS.

PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i) over bins. Numeric bins are the reference deciles
(duplicated edges collapse for tied / discrete features) with open outer edges, plus one bin for
missing values, so a change in missingness also counts as drift. Proportions are floored at `eps`
so empty bins stay finite. PSI cut-offs (~0.10 / ~0.25) are rules of thumb, not decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import config

MISSING_BIN = "missing"


def quantile_edges(reference: pd.Series, n_bins: int = config.PSI_BINS) -> np.ndarray:
    """Interior bin edges = unique reference quantiles (learned from the reference only)."""
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return np.array([])
    interior = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(interior)


def _numeric_bin_labels(values: pd.Series, edges: np.ndarray) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    bins = np.concatenate([[-np.inf], edges, [np.inf]])
    labels = pd.cut(values, bins=bins, right=True, labels=False)
    labels = labels.astype("Int64").astype(str)
    return labels.where(values.notna(), MISSING_BIN)


def _psi_from_labels(ref_labels: pd.Series, cur_labels: pd.Series, bin_order: list, eps: float) -> tuple[float, pd.DataFrame]:
    ref_pct = ref_labels.value_counts(normalize=True).reindex(bin_order, fill_value=0.0)
    cur_pct = cur_labels.value_counts(normalize=True).reindex(bin_order, fill_value=0.0)
    ref_c, cur_c = ref_pct.clip(lower=eps), cur_pct.clip(lower=eps)
    contrib = (cur_c - ref_c) * np.log(cur_c / ref_c)
    table = pd.DataFrame({"bin": bin_order, "ref_pct": ref_pct.to_numpy(), "cur_pct": cur_pct.to_numpy(),
                          "psi_contribution": contrib.to_numpy()})
    return float(contrib.sum()), table


def psi_numeric(reference: pd.Series, current: pd.Series, n_bins: int = config.PSI_BINS,
                eps: float = config.PSI_EPS) -> tuple[float, pd.DataFrame]:
    edges = quantile_edges(reference, n_bins)
    bin_order = [str(i) for i in range(len(edges) + 1)] + [MISSING_BIN]
    value, table = _psi_from_labels(_numeric_bin_labels(reference, edges),
                                    _numeric_bin_labels(current, edges), bin_order, eps)
    lower = np.concatenate([[-np.inf], edges])
    upper = np.concatenate([edges, [np.inf]])
    table["interval"] = [f"({lo:g}, {hi:g}]" for lo, hi in zip(lower, upper)] + [MISSING_BIN]
    return value, table


def psi_categorical(reference: pd.Series, current: pd.Series, eps: float = config.PSI_EPS) -> tuple[float, pd.DataFrame]:
    """Categories come from the reference; unseen categories in `current` share an "unseen" bin."""
    ref = reference.astype(object).where(reference.notna(), MISSING_BIN)
    cur = current.astype(object).where(current.notna(), MISSING_BIN)
    categories = sorted(ref.unique().tolist(), key=str)
    cur = cur.where(cur.isin(categories), "unseen")
    return _psi_from_labels(ref, cur, categories + ["unseen"], eps)


def ks_drift(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Two-sample KS on observed (non-missing) values: statistic and p-value."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return np.nan, np.nan
    result = ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def psi_band(value: float) -> str:
    low, high = config.PSI_REFERENCE_CUTOFFS
    if np.isnan(value):
        return "n/a"
    return "pouca mudança" if value < low else ("atenção" if value < high else "mudança relevante")


def numeric_drift_table(reference: pd.DataFrame, current: pd.DataFrame, features: list[str],
                        reference_name: str, current_name: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        psi_value, _ = psi_numeric(reference[feature], current[feature])
        ks_stat, ks_p = ks_drift(reference[feature], current[feature])
        rows.append({
            "reference": reference_name,
            "current": current_name,
            "feature": feature,
            "psi": psi_value,
            "psi_band_heuristic": psi_band(psi_value),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "ref_n": int(reference[feature].notna().sum()),
            "cur_n": int(current[feature].notna().sum()),
            "ref_missing_rate": float(reference[feature].isna().mean()),
            "cur_missing_rate": float(current[feature].isna().mean()),
            "ref_median": float(reference[feature].median()),
            "cur_median": float(current[feature].median()),
            "ref_mean": float(reference[feature].mean()),
            "cur_mean": float(current[feature].mean()),
        })
    return pd.DataFrame(rows)


def positive_rate(target: pd.Series) -> float:
    """Share of RainTomorrow == "Yes" among labelled rows."""
    labelled = target.dropna()
    return float((labelled == config.POSITIVE_LABEL).mean()) if len(labelled) else np.nan
