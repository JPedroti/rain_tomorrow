"""Figures shared by the scripts and the notebook (matplotlib, static PNG, light surface).

Colour roles: at most three categorical series per chart (validated reference slots blue, orange,
aqua); a blue <-> red diverging pair for signed coefficients; grey ink for text and references.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.labelcolor": INK_2, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def _finish(fig, path: Path | str | None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_roc_pr(curves: dict[str, tuple], title: str = "", path=None):
    """ROC and Precision-Recall curves for up to three labelled (y_true, y_score) pairs."""
    apply_style()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    prevalences = {label: float(np.mean(y)) for label, (y, _) in curves.items()}
    top = max(prevalences.values())
    for color, (label, (y, s)) in zip(SERIES, curves.items()):
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        ax_roc.plot(fpr, tpr, color=color, label=label)
        ax_pr.plot(rec, prec, color=color, label=label, drawstyle="steps-post")
        prevalence = prevalences[label]
        above = prevalence == top
        ax_pr.axhline(prevalence, color=color, linewidth=1, alpha=0.6)
        ax_pr.annotate(f"prevalência {label}: {prevalence:.3f}", xy=(0.0, prevalence), xytext=(4, 3 if above else -3),
                       textcoords="offset points", ha="left", va="bottom" if above else "top", color=INK_2, fontsize=8)
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax_roc.annotate("ordenação aleatória", xy=(0.62, 0.58), color=MUTED, fontsize=8, rotation=38)
    ax_roc.set(xlabel="Taxa de falsos positivos (1 − especificidade)", ylabel="Taxa de verdadeiros positivos (recall)",
               title="Curva ROC", xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall", xlim=(0, 1), ylim=(0, 1.01))
    if len(curves) > 1:
        ax_roc.legend(loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path)


def plot_score_distributions(y, score, title: str = "", path=None):
    """Score histograms by class and the two ECDFs with the KS gap marked."""
    apply_style()
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    pos, neg = np.sort(score[y == 1]), np.sort(score[y == 0])
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(11, 4.4))
    bins = np.linspace(0, 1, 41)
    for values, color, label in [(neg, SERIES[1], "RainTomorrow = No"), (pos, SERIES[0], "RainTomorrow = Yes")]:
        ax_h.hist(values, bins=bins, density=True, color=color, alpha=0.12)
        ax_h.hist(values, bins=bins, density=True, histtype="step", color=color, linewidth=2, label=label)
    ax_h.set(xlabel="Score (probabilidade prevista de chuva)", ylabel="Densidade", title="Distribuição dos scores por classe")
    ax_h.legend()

    grid = np.linspace(0, 1, 1001)
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    gap = cdf_neg - cdf_pos
    k = int(np.argmax(gap))
    ax_c.plot(grid, cdf_neg, color=SERIES[1], label="RainTomorrow = No")
    ax_c.plot(grid, cdf_pos, color=SERIES[0], label="RainTomorrow = Yes")
    ax_c.vlines(grid[k], cdf_pos[k], cdf_neg[k], color=INK, linewidth=1.5)
    ax_c.annotate(f"KS = {gap[k]:.3f}\n(score = {grid[k]:.2f})", xy=(grid[k], (cdf_pos[k] + cdf_neg[k]) / 2),
                  xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax_c.set(xlabel="Score", ylabel="Proporção acumulada", title="Distribuições acumuladas (KS)", xlim=(0, 1), ylim=(0, 1.01))
    ax_c.legend(loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path)


def plot_cv_path(cv_results: pd.DataFrame, path=None):
    """Stage-2 CV ROC-AUC (mean ± 1 SE) against C, one panel per class_weight, L2 vs L1."""
    apply_style()
    stage2 = cv_results[cv_results["stage"] == "2_hyperparameters"].copy()
    stage2["class_weight"] = stage2["class_weight"].fillna("None")
    panels = ["None", "balanced"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cw in zip(axes, panels):
        sub = stage2[stage2["class_weight"] == cw]
        for color, pen in zip(SERIES, ["l2", "l1"]):
            s = sub[sub["penalty"] == pen].sort_values("C")
            ax.errorbar(s["C"], s["roc_auc_mean"], yerr=s["roc_auc_se"], color=color, marker="o",
                        markersize=5, capsize=3, label=pen.upper())
        chosen = sub[sub["selected"]]
        if not chosen.empty:
            ax.scatter(chosen["C"], chosen["roc_auc_mean"], s=180, facecolors="none", edgecolors=INK, linewidths=1.5,
                       zorder=5, label="selecionado (regra 1-SE)")
        ax.set_xscale("log")
        ax.set(xlabel="C (inverso da força de regularização)", title=f"class_weight = {cw}")
    axes[0].set_ylabel("ROC-AUC médio na CV (± 1 erro-padrão)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return _finish(fig, path)


def plot_coefficients(coef_table: pd.DataFrame, top_n: int = 20, path=None):
    """Largest standardised coefficients; blue raises, red lowers the log-odds of rain."""
    apply_style()
    top = coef_table.head(top_n).iloc[::-1]
    colors = [POSITIVE if c > 0 else NEGATIVE for c in top["coefficient"]]
    fig, ax = plt.subplots(figsize=(8, 0.32 * top_n + 1.2))
    ax.barh(top["feature"], top["coefficient"], color=colors, height=0.6)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set(xlabel="Coeficiente (log-odds por desvio-padrão / categoria)", title=f"{top_n} maiores coeficientes (em módulo)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _finish(fig, path)


def plot_monthly_performance(monthly: pd.DataFrame, reference: pd.DataFrame | None = None,
                             reference_label: str = "", path=None):
    """Small multiples: ROC-AUC, PR-AUC, KS (with bootstrap CI) and the positive rate per month.

    `monthly` rows are OOT months; `reference` (optional) holds the same metrics for the same calendar
    months in the historical TEST set, joined on a `month` column (1-12).
    """
    apply_style()
    panels = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("ks", "KS"), ("prevalence", "Taxa de RainTomorrow = Yes")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    x = np.arange(len(monthly))
    for ax, (metric, label) in zip(axes.ravel(), panels):
        ax.plot(x, monthly[metric], color=SERIES[0], marker="o", markersize=6, label="OOT 2017")
        if f"{metric}_ci_low" in monthly:
            ax.fill_between(x, monthly[f"{metric}_ci_low"], monthly[f"{metric}_ci_high"], color=SERIES[0], alpha=0.10,
                            linewidth=0, label="IC 95% (bootstrap)")
        if reference is not None and metric in reference:
            ref = monthly[["month"]].merge(reference, on="month", how="left")
            ax.plot(x, ref[metric], color=SERIES[1], marker="o", markersize=6, label=reference_label)
        ax.set_title(label)
    for ax in axes[-1]:
        ax.set_xticks(x, monthly["period"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    return _finish(fig, path)


def plot_drift(drift: pd.DataFrame, path=None):
    """PSI and KS statistic per feature for each reference (horizontal grouped bars)."""
    apply_style()
    references = list(drift["reference"].unique())
    order = (drift[drift["reference"] == references[0]].sort_values("psi")["feature"].tolist())
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.34 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))
    height = 0.8 / len(references)
    for i, (ref, color) in enumerate(zip(references, SERIES)):
        sub = drift[drift["reference"] == ref].set_index("feature").loc[order]
        offset = (i - (len(references) - 1) / 2) * height
        axes[0].barh(y + offset, sub["psi"], height=height * 0.9, color=color, label=ref)
        axes[1].barh(y + offset, sub["ks_statistic"], height=height * 0.9, color=color, label=ref)
    for cut in (0.10, 0.25):
        axes[0].axvline(cut, color=MUTED, linewidth=1)
        axes[0].annotate(f"{cut:.2f}", xy=(cut, len(order) - 0.4), xytext=(3, 0), textcoords="offset points",
                         color=MUTED, fontsize=8)
    axes[0].set_yticks(y, order)
    axes[0].set(xlabel="PSI (bins = decis do TRAIN + bin de missing)", title="PSI por feature")
    axes[1].set(xlabel="Estatística KS (duas amostras)", title="KS de drift por feature")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _finish(fig, path)


def plot_target_rate(monthly_rate: pd.DataFrame, path=None):
    """Positive rate by calendar month: historical development years vs OOT 2017."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for color, col, label in [(SERIES[1], "train_rate", "TRAIN < 2017 (mesmo mês)"), (SERIES[0], "oot_rate", "OOT 2017")]:
        sub = monthly_rate.dropna(subset=[col])
        ax.plot(sub["month"], sub[col], color=color, marker="o", markersize=6, label=label)
    ax.set_xticks(range(1, 13))
    ax.set(xlabel="Mês do ano", ylabel="Taxa de RainTomorrow = Yes", title="Target drift: taxa de chuva por mês")
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path)
