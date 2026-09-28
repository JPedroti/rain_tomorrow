"""Discrimination metrics used everywhere in the project (same procedure for every set and window).

    ROC-AUC : sklearn roc_auc_score
    PR-AUC  : sklearn average_precision_score (step-wise Average Precision, no trapezoidal interpolation)
    KS      : max(TPR - FPR) over the ROC curve == two-sample KS statistic between the score
              distributions of positives and negatives (0-1 scale)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from src import config

DISCRIMINATION_METRICS = ("roc_auc", "pr_auc", "ks")


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def discrimination_metrics(
    y_true,
    y_score,
    min_rows: int = 1,
    min_per_class: int = 1,
) -> dict:
    """ROC-AUC, PR-AUC and KS with counts. Returns NaN metrics and a reason when the sample is unusable."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n, n_pos = len(y_true), int(y_true.sum())
    out = {"n": n, "n_pos": n_pos, "n_neg": n - n_pos, "prevalence": n_pos / n if n else np.nan}
    if n < min_rows or n_pos < min_per_class or (n - n_pos) < min_per_class:
        reason = "single_class" if n_pos in (0, n) else "insufficient_volume"
        return {**out, **dict.fromkeys(DISCRIMINATION_METRICS, np.nan), "status": reason}
    return {
        **out,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "status": "ok",
    }


def bootstrap_ci(
    y_true,
    y_score,
    n_boot: int = config.N_BOOTSTRAP,
    level: float = config.CI_LEVEL,
    seed: int = config.RANDOM_STATE,
) -> dict:
    """Percentile bootstrap (rows resampled with replacement) for the three metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = {m: [] for m in DISCRIMINATION_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.min() == yt.max():
            continue
        draws["roc_auc"].append(roc_auc_score(yt, ys))
        draws["pr_auc"].append(average_precision_score(yt, ys))
        draws["ks"].append(ks_statistic(yt, ys))
    tail = (1.0 - level) / 2.0
    return {
        m: (float(np.quantile(v, tail)), float(np.quantile(v, 1.0 - tail))) if v else (np.nan, np.nan)
        for m, v in draws.items()
    }


def evaluate_scores(y_true, y_score, with_ci: bool = True, **ci_kwargs) -> dict:
    """Metrics + bootstrap intervals flattened as <metric>, <metric>_ci_low, <metric>_ci_high."""
    result = discrimination_metrics(y_true, y_score)
    if with_ci and result["status"] == "ok":
        for metric, (low, high) in bootstrap_ci(y_true, y_score, **ci_kwargs).items():
            result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = low, high
    return result


def metrics_by_group(
    keys,
    y_true,
    y_score,
    key_name: str = "group",
    min_rows: int = config.MIN_WINDOW_ROWS,
    min_per_class: int = config.MIN_WINDOW_PER_CLASS,
    with_ci: bool = True,
    n_boot: int = config.N_BOOTSTRAP,
) -> pd.DataFrame:
    """Metrics for each group. Groups failing the volume rule get NaN metrics + a status flag."""
    frame = pd.DataFrame({
        key_name: np.asarray(keys),
        "y": np.asarray(y_true).astype(int),
        "score": np.asarray(y_score, dtype=float),
    })
    rows = []
    for key, grp in frame.groupby(key_name, sort=True):
        res = discrimination_metrics(grp["y"], grp["score"], min_rows, min_per_class)
        if with_ci and res["status"] == "ok":
            for metric, (low, high) in bootstrap_ci(grp["y"], grp["score"], n_boot=n_boot).items():
                res[f"{metric}_ci_low"], res[f"{metric}_ci_high"] = low, high
        res["mean_score"] = float(grp["score"].mean())
        rows.append({key_name: key, **res})
    return pd.DataFrame(rows)


def metrics_by_period(dates: pd.Series, y_true, y_score, freq: str = "M", **kwargs) -> pd.DataFrame:
    """Metrics for each calendar window (default: month, e.g. "2017-03")."""
    periods = pd.to_datetime(pd.Series(dates)).dt.to_period(freq).astype(str)
    return metrics_by_group(periods, y_true, y_score, key_name="period", **kwargs)


def threshold_metrics(y_true, y_score, threshold: float) -> dict:
    """Confusion matrix and threshold-dependent metrics at one ILLUSTRATIVE threshold (didactic only)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": 2 * precision * recall / (precision + recall) if precision and recall else np.nan,
        "predicted_positive_rate": float(y_pred.mean()),
    }
