"""Evaluate the FROZEN model on the Out-of-Time period (2017) - run only after scripts/train.py.

    python scripts/evaluate.py

Loads artifacts/model.joblib after checking its SHA-256 against artifacts/metadata.json and never
calls `fit`. Produces:
    artifacts/metrics.json            (adds OOT, best/worst month and the final comparison)
    reports/temporal_performance.csv  (monthly OOT metrics + same calendar month in the historical TEST)
    reports/target_drift.csv          (positive rate: TRAIN / TEST / OOT and by calendar month)
    reports/drift_report.csv          (PSI + KS per feature, TRAIN -> OOT, two reference windows)
    reports/figures/oot_*.png, target_rate_by_month.png, feature_drift.png
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src import config, plots  # noqa: E402
from src.artifacts import load_frozen_model, read_json, write_json  # noqa: E402
from src.data import drop_missing_target, file_sha256, get_xy, load_oot_data, load_train_test  # noqa: E402
from src.evaluation import METRIC_KEYS, feature_drift_report, monthly_performance, target_drift_report  # noqa: E402
from src.metrics import discrimination_metrics, evaluate_scores  # noqa: E402
from src.modeling import persistence_score  # noqa: E402

def main() -> None:
    model, metadata = load_frozen_model()
    frozen_sha = metadata["model_sha256"]
    print(f"frozen model loaded, sha256 verified: {frozen_sha}")
    metrics = read_json(config.METRICS_PATH)

    # Development sets rebuilt deterministically (only transform / predict_proba below).
    data = load_train_test()
    train, test = data["train"], data["test"]
    assert len(train) == metadata["temporal_protocol"]["train_rows"]
    assert len(test) == metadata["temporal_protocol"]["test_rows"]
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)
    s_train = model.predict_proba(X_train)[:, 1]
    s_test = model.predict_proba(X_test)[:, 1]
    check = discrimination_metrics(y_test, s_test)
    for m in METRIC_KEYS:
        assert abs(check[m] - metrics["test"][m]) < 1e-12, f"TEST {m} differs from the value recorded at freeze"

    # ---- OOT opened here ----------------------------------------------------------------------------
    opened_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    oot_all = load_oot_data()
    oot, n_missing = drop_missing_target(oot_all)
    X_oot, y_oot = get_xy(oot)
    s_oot = model.predict_proba(X_oot)[:, 1]
    # Every OOT row is scored for score drift (the target is not needed to score).
    s_oot_all = model.predict_proba(oot_all.drop(columns=[config.TARGET_COL]))[:, 1]
    oot_metrics = evaluate_scores(y_oot, s_oot)
    persistence_oot = discrimination_metrics(y_oot, persistence_score(X_oot))
    print(f"OOT: {len(oot_all)} rows, {n_missing} without target, {len(oot)} evaluated")
    print(f"OOT : ROC-AUC {oot_metrics['roc_auc']:.4f} | PR-AUC {oot_metrics['pr_auc']:.4f} | KS {oot_metrics['ks']:.4f}")

    monthly = monthly_performance(oot, y_oot, s_oot, test, y_test, s_test)
    monthly.to_csv(config.TEMPORAL_PERFORMANCE_PATH, index=False)
    valid = monthly[monthly["status"] == "ok"]
    best = valid.loc[valid[config.PERIOD_RANKING_METRIC].idxmax()]
    worst = valid.loc[valid[config.PERIOD_RANKING_METRIC].idxmin()]

    by_month, rates = target_drift_report(train, test, oot)
    by_month.to_csv(config.TARGET_DRIFT_PATH, index=False)
    drift = feature_drift_report(train, oot_all, s_train, s_oot_all)
    drift.to_csv(config.DRIFT_REPORT_PATH, index=False)

    # Figures
    plots.plot_roc_pr({"TEST": (y_test, s_test), "OOT 2017": (y_oot, s_oot)},
                      title="Modelo congelado: TEST (histórico) vs OOT 2017", path=config.FIGURES_DIR / "oot_vs_test_roc_pr.png")
    plots.plot_score_distributions(y_oot, s_oot, title="Separação dos scores no OOT 2017",
                                   path=config.FIGURES_DIR / "oot_score_distribution.png")
    ref_cols = {f"hist_test_{m}": m for m in [*METRIC_KEYS, "prevalence"]}
    plots.plot_monthly_performance(monthly, monthly[["month", *ref_cols]].rename(columns=ref_cols),
                                   reference_label="TEST histórico (mesmo mês)",
                                   path=config.FIGURES_DIR / "oot_monthly_performance.png")
    plots.plot_target_rate(by_month, path=config.FIGURES_DIR / "target_rate_by_month.png")
    plots.plot_drift(drift[drift["type"] == "numérica"], path=config.FIGURES_DIR / "feature_drift.png")

    def row(label, res, note=""):
        return {"set": label, **{m: res[m] for m in METRIC_KEYS}, "n": int(res["n"]),
                "prevalence": float(res["prevalence"]), "note": note}

    metrics["oot"] = {**oot_metrics, "n_rows_total": len(oot_all), "n_missing_target": n_missing}
    metrics["test_to_oot_delta"] = {m: oot_metrics[m] - metrics["test"][m] for m in (*METRIC_KEYS, "prevalence")}
    metrics["references_oot"] = {"no_skill": {"roc_auc": 0.5, "pr_auc": float(y_oot.mean()), "ks": 0.0},
                                 "persistence_rain_today": persistence_oot}
    metrics["oot_months"] = {
        "ranking_metric": config.PERIOD_RANKING_METRIC,
        "window_rule": f"janela avaliada só com >= {config.MIN_WINDOW_ROWS} linhas e >= {config.MIN_WINDOW_PER_CLASS} "
                       "casos de cada classe; caso contrário métricas vazias com status",
        "best": best.to_dict(), "worst": worst.to_dict(),
        "n_months": len(monthly), "n_months_evaluated": len(valid),
        "insufficient_windows": monthly.loc[monthly["status"] != "ok", ["period", "n", "status"]].to_dict("records"),
    }
    metrics["target_rates"] = rates
    rule = f"critério: {config.PERIOD_RANKING_METRIC} entre meses com volume suficiente"
    metrics["final_comparison"] = [
        row("Teste", metrics["test"], "20% aleatório estratificado do histórico < 2017"),
        row("OOT 2017", oot_metrics, "todas as linhas de 2017 com target (jan-jun)"),
        row(f"Melhor mês do OOT ({best['period']})", best, f"maior {rule}"),
        row(f"Pior mês do OOT ({worst['period']})", worst, f"menor {rule}"),
    ]
    metrics["oot_status"] = "avaliado após o congelamento"
    metrics["oot_evaluated_at_utc"] = opened_at
    write_json(config.METRICS_PATH, metrics)

    after_sha = file_sha256(config.MODEL_PATH)
    assert after_sha == frozen_sha, "model file changed during evaluation"
    # Append-only evaluation record; the freeze fields of metadata.json are left untouched.
    metadata["oot_evaluation"] = {
        "evaluated_at_utc": opened_at,
        "model_sha256_verified_before": frozen_sha,
        "model_sha256_after": after_sha,
        "fit_called": False,
        "oot_rows_total": len(oot_all), "oot_rows_with_target": len(oot), "oot_rows_missing_target": n_missing,
        "oot_date_range": [str(oot_all["Date"].min().date()), str(oot_all["Date"].max().date())],
        "reports": ["reports/temporal_performance.csv", "reports/target_drift.csv", "reports/drift_report.csv"],
    }
    write_json(config.METADATA_PATH, metadata)
    print(f"best month {best['period']} ROC-AUC {best['roc_auc']:.4f} | worst month {worst['period']} ROC-AUC {worst['roc_auc']:.4f}")
    print("model hash unchanged after evaluation")


if __name__ == "__main__":
    main()
