"""Post-freeze analyses of the OOT period: monthly stability, target drift and feature drift.

Only `predict_proba` outputs of the frozen model enter here; nothing in this module fits anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.drift import ks_drift, numeric_drift_table, positive_rate, psi_band, psi_categorical, psi_numeric
from src.metrics import metrics_by_group, metrics_by_period

METRIC_KEYS = ("roc_auc", "pr_auc", "ks")


def monthly_performance(oot, y_oot, s_oot, test, y_test, s_test) -> pd.DataFrame:
    """OOT metrics per month, side by side with the same calendar month of the historical TEST."""
    monthly = metrics_by_period(oot["Date"], y_oot, s_oot)
    monthly["month"] = pd.PeriodIndex(monthly["period"], freq="M").month
    hist = metrics_by_group(test["Date"].dt.month.to_numpy(), y_test, s_test, key_name="month", with_ci=False)
    hist = hist[["month", "n", "prevalence", *METRIC_KEYS]].rename(
        columns={c: f"hist_test_{c}" for c in ["n", "prevalence", *METRIC_KEYS]})
    monthly = monthly.merge(hist, on="month", how="left")
    for m in METRIC_KEYS:
        monthly[f"delta_{m}_vs_hist_test"] = monthly[m] - monthly[f"hist_test_{m}"]
    return monthly


def target_drift_report(train, test, oot) -> tuple[pd.DataFrame, dict]:
    """Positive rate overall (TRAIN/TEST/OOT) and by calendar month (TRAIN vs OOT)."""
    rates = {"train": positive_rate(train[config.TARGET_COL]), "test": positive_rate(test[config.TARGET_COL]),
             "oot": positive_rate(oot[config.TARGET_COL])}
    by_month = pd.DataFrame({"month": range(1, 13)})
    tr = train.groupby(train["Date"].dt.month)[config.TARGET_COL].agg(dev_n="size", dev_rate=positive_rate)
    oo = oot.dropna(subset=[config.TARGET_COL]).groupby(oot["Date"].dt.month)[config.TARGET_COL].agg(
        oot_n="size", oot_rate=positive_rate)
    by_month = by_month.merge(tr, left_on="month", right_index=True, how="left")
    by_month = by_month.merge(oo, left_on="month", right_index=True, how="left")
    by_month["oot_minus_dev"] = by_month["oot_rate"] - by_month["dev_rate"]
    # Rate expected in the OOT if each month kept its historical TRAIN rate (same month mix as the OOT).
    w = by_month.dropna(subset=["oot_n"])
    rates["oot_expected_from_train_seasonality"] = float(np.average(w["dev_rate"], weights=w["oot_n"]))
    rates["train_jan_jun"] = positive_rate(train.loc[train["Date"].dt.month <= 6, config.TARGET_COL])
    return by_month, rates


def feature_drift_report(train, oot, train_scores, oot_scores) -> pd.DataFrame:
    """PSI/KS TRAIN -> OOT with two references: all TRAIN months, and TRAIN Jan-Jun (same season)."""
    season = train["Date"].dt.month.isin(sorted(oot["Date"].dt.month.unique()))
    references = {"TRAIN (todos os meses)": (train, train_scores),
                  "TRAIN (mesmos meses do OOT: jan-jun)": (train.loc[season], train_scores[season.to_numpy()])}
    frames = []
    for name, (ref, ref_scores) in references.items():
        num = numeric_drift_table(ref, oot, config.NUMERIC_FEATURES, name, "OOT 2017")
        num.insert(3, "type", "numérica")
        frames.append(num)
        psi_loc, _ = psi_categorical(ref[config.LOCATION_COL], oot[config.LOCATION_COL])
        frames.append(pd.DataFrame([{"reference": name, "current": "OOT 2017", "feature": config.LOCATION_COL,
                                     "type": "categórica", "psi": psi_loc, "psi_band_heuristic": psi_band(psi_loc)}]))
        psi_s, _ = psi_numeric(pd.Series(ref_scores), pd.Series(oot_scores))
        ks_s, ks_p = ks_drift(pd.Series(ref_scores), pd.Series(oot_scores))
        frames.append(pd.DataFrame([{"reference": name, "current": "OOT 2017", "feature": "score do modelo",
                                     "type": "score", "psi": psi_s, "psi_band_heuristic": psi_band(psi_s),
                                     "ks_statistic": ks_s, "ks_pvalue": ks_p,
                                     "ref_median": float(np.median(ref_scores)), "cur_median": float(np.median(oot_scores)),
                                     "ref_mean": float(np.mean(ref_scores)), "cur_mean": float(np.mean(oot_scores))}]))
    return pd.concat(frames, ignore_index=True)
