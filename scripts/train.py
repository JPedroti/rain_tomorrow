"""Train, select and freeze the Rain Tomorrow model using development data only (Date < 2017-01-01).

    python scripts/train.py [--n-jobs N]

1. Development rows only (the OOT is discarded right after loading) -> drop rows without target ->
   stratified 80/20 TRAIN/TEST split (random_state=42).
2. Baseline: simple pipeline + default LogisticRegression, registered on TEST.
3. Pre-registered selection with 5-fold stratified CV on TRAIN (stage 1: preprocessing; stage 2:
   C / penalty / class_weight), one-standard-error rule on ROC-AUC.
4. Final pipeline fitted on TRAIN -> TRAIN and TEST metrics (bootstrap CI on TEST).
5. Informative sensitivity analysis across feature sets (never used to choose the frozen model).
6. Freeze: model.joblib + SHA-256 in metadata.json; metrics.json; reports and figures.

This script never reads the OOT rows. The OOT is evaluated afterwards by scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, plots  # noqa: E402
from src.artifacts import environment_info, save_model, source_hashes, write_json  # noqa: E402
from src.data import assert_no_oot, file_sha256, get_xy, load_train_test  # noqa: E402
from src.metrics import discrimination_metrics, evaluate_scores, threshold_metrics  # noqa: E402
from src.modeling import (  # noqa: E402
    build_pipeline,
    persistence_score,
    coefficient_table,
    evaluate_candidates,
    penalty_name,
    run_model_selection,
    solver_for,
)
from src.preprocessing import input_columns  # noqa: E402

PREDICTION_MOMENT = (
    "Previsão emitida ao fim do dia t (00:00 de t+1), com as observações do dia t. O alvo é a chuva > 1 mm "
    "nas 24h entre 09:00 de t e 09:00 de t+1; as observações das 15h, Sunshine e WindGust* caem dentro dessa "
    "janela, mas fecham até 00:00 de t+1. MaxTemp foi excluída: sua janela (09:00 de t a 09:00 de t+1) só "
    "fecha depois do momento da previsão (ver FEATURE_WINDOWS)."
)
SELECTION_PROTOCOL = (
    "CV estratificada 5-fold somente no TRAIN. Estágio 1: variantes de preprocessing com a LR baseline. "
    "Estágio 2: grade C x penalty (l1_ratio 0/1) x class_weight na variante escolhida. Métrica: ROC-AUC "
    "médio. Regra de 1 erro-padrão: entre os candidatos com média >= melhor média - EP do melhor, escolhe o "
    "primeiro na ordem de preferência (preprocessing mais simples; class_weight=None, menor C, L2 antes de L1)."
)


def sensitivity_analysis(params: dict, train: pd.DataFrame, y_train, test: pd.DataFrame, y_test, n_jobs: int) -> pd.DataFrame:
    """Same frozen configuration, only the feature set changes. Informative only."""
    base = {k: v for k, v in params.items() if k != "feature_set"}
    sets = list(config.FEATURE_SETS)
    cv = evaluate_candidates([{"feature_set": fs, **base} for fs in sets], train, y_train, n_jobs=n_jobs)
    rows = []
    for fs, (_, cv_row) in zip(sets, cv.iterrows()):
        pipe = build_pipeline(feature_set=fs, **base).fit(train, y_train)
        test_m = discrimination_metrics(y_test, pipe.predict_proba(test)[:, 1])
        spec = config.FEATURE_SETS[fs]
        rows.append({
            "feature_set": fs,
            "is_final_model": fs == params["feature_set"],
            "n_raw_features": len(spec["numeric"]) + len(spec["categorical"]),
            "features": ", ".join(spec["numeric"] + spec["categorical"]),
            **{f"cv_{m}_{s}": cv_row[f"{m}_{s}"] for m in ("roc_auc", "pr_auc", "ks") for s in ("mean", "se")},
            **{f"test_{m}": test_m[m] for m in ("roc_auc", "pr_auc", "ks")},
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for cross-validation")
    args = parser.parse_args()
    started = time.perf_counter()

    for directory in (config.ARTIFACTS_DIR, config.REPORTS_DIR, config.FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_sha = file_sha256(config.DATA_PATH)
    if dataset_sha != config.DATASET_SHA256:
        print(f"WARNING: dataset sha256 {dataset_sha} differs from the audited file {config.DATASET_SHA256}")

    # 1. Development data only --------------------------------------------------------------------
    data = load_train_test()
    train, test = data["train"], data["test"]
    assert_no_oot(train)
    assert_no_oot(test)
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)
    print(f"development rows: {data['n_dev_rows']} | dropped (missing target): {data['n_dropped_missing_target']}")
    print(f"TRAIN: {len(train)} rows, prevalence {y_train.mean():.4f} | TEST: {len(test)} rows, prevalence {y_test.mean():.4f}")

    # 2. Baseline -------------------------------------------------------------------------------------
    baseline_params = {"feature_set": config.FINAL_FEATURE_SET, "preprocessing": config.BASELINE_PREPROCESSING,
                       **config.BASELINE_MODEL_PARAMS}
    baseline = build_pipeline(**baseline_params).fit(X_train, y_train)
    baseline_test = evaluate_scores(y_test, baseline.predict_proba(X_test)[:, 1])
    print(f"baseline TEST: ROC-AUC {baseline_test['roc_auc']:.4f} | PR-AUC {baseline_test['pr_auc']:.4f} | KS {baseline_test['ks']:.4f}")

    # 3. Model selection on TRAIN (CV) ------------------------------------------------------------------
    selection = run_model_selection(X_train, y_train, feature_set=config.FINAL_FEATURE_SET, n_jobs=args.n_jobs)
    params = selection["selected_params"]
    cv_results = selection["cv_results"]
    cv_results.to_csv(config.CV_RESULTS_PATH, index=False)
    selected_row = cv_results[(cv_results["stage"] == "2_hyperparameters") & cv_results["selected"]].iloc[0]
    print(f"selected: {params}")

    # 4. Final fit on TRAIN, metrics on TRAIN and TEST ---------------------------------------------------
    model = build_pipeline(**params).fit(X_train, y_train)
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    train_metrics = discrimination_metrics(y_train, train_scores)
    test_metrics = evaluate_scores(y_test, test_scores)
    persistence_test = discrimination_metrics(y_test, persistence_score(X_test))
    print(f"final TRAIN: ROC-AUC {train_metrics['roc_auc']:.4f} | PR-AUC {train_metrics['pr_auc']:.4f} | KS {train_metrics['ks']:.4f}")
    print(f"final TEST : ROC-AUC {test_metrics['roc_auc']:.4f} | PR-AUC {test_metrics['pr_auc']:.4f} | KS {test_metrics['ks']:.4f}")

    # 5. Informative analyses (never feed back into the choice above) -----------------------------------
    sensitivity = sensitivity_analysis(params, X_train, y_train, X_test, y_test, args.n_jobs)
    sensitivity.to_csv(config.SENSITIVITY_PATH, index=False)
    thresholds = pd.DataFrame([threshold_metrics(y_test, test_scores, t) for t in config.ILLUSTRATIVE_THRESHOLDS])
    thresholds.to_csv(config.REPORTS_DIR / "threshold_examples_test.csv", index=False)
    coefficients = coefficient_table(model)
    coefficients.to_csv(config.REPORTS_DIR / "coefficients.csv", index=False)

    plots.plot_roc_pr({"TEST": (y_test, test_scores)}, title="Modelo final no TEST (histórico < 2017)",
                      path=config.FIGURES_DIR / "test_roc_pr.png")
    plots.plot_score_distributions(y_test, test_scores, title="Separação dos scores no TEST",
                                   path=config.FIGURES_DIR / "test_score_distribution.png")
    plots.plot_cv_path(cv_results, path=config.FIGURES_DIR / "cv_regularization_path.png")
    plots.plot_coefficients(coefficients, path=config.FIGURES_DIR / "coefficients.png")

    # 6. Freeze -------------------------------------------------------------------------------------------
    model_sha = save_model(model, config.MODEL_PATH)
    spec = config.FEATURE_SETS[params["feature_set"]]
    lr = model.named_steps["model"]
    metadata = {
        "project": "Rain Tomorrow - Projeto 04",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": "artifacts/model.joblib",
        "model_sha256": model_sha,
        "dataset": {"path": "data/weatherAUS.csv", "sha256": dataset_sha, "audited_sha256": config.DATASET_SHA256},
        "target": {"column": config.TARGET_COL, "positive_class": config.POSITIVE_LABEL,
                   "definition": "Chuva > 1 mm nas 24h entre 09:00 de t e 09:00 de t+1 (= RainToday de t+1)"},
        "temporal_protocol": {
            "oot_start": config.OOT_START,
            "development_rows": data["n_dev_rows"],
            "dropped_missing_target": data["n_dropped_missing_target"],
            "train_rows": len(train), "test_rows": len(test),
            "train_date_range": [str(train["Date"].min().date()), str(train["Date"].max().date())],
            "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE, "stratify": True,
            "oot_read_by_training": False,
        },
        "features": {
            "feature_set": params["feature_set"],
            "numeric": spec["numeric"], "categorical": spec["categorical"],
            "date_features": ["doy_sin", "doy_cos"],
            "input_columns": input_columns(params["feature_set"]),
            "n_model_columns": int(len(model.named_steps["preprocess"].get_feature_names_out())),
            "prediction_moment": PREDICTION_MOMENT,
            "measurement_windows": {f: config.FEATURE_WINDOWS[f] for f in spec["numeric"] + spec["categorical"]},
        },
        "model": {
            "algorithm": "sklearn.linear_model.LogisticRegression",
            "preprocessing_variant": params["preprocessing"],
            "preprocessing_options": config.PREPROCESSING_VARIANTS[params["preprocessing"]],
            "C": params["C"], "l1_ratio": params["l1_ratio"], "penalty": penalty_name(params["l1_ratio"]),
            "solver": solver_for(params["l1_ratio"]), "class_weight": params["class_weight"],
            "max_iter": config.MAX_ITER, "n_iter": lr.n_iter_.tolist(), "intercept": float(lr.intercept_[0]),
            "n_nonzero_coefficients": int(np.count_nonzero(lr.coef_)),
        },
        "selection": {
            "protocol": SELECTION_PROTOCOL, "metric": config.SELECTION_METRIC, "cv_folds": config.CV_FOLDS,
            "C_grid": config.C_GRID, "l1_ratio_grid": config.L1_RATIO_GRID,
            "class_weight_grid": config.CLASS_WEIGHT_GRID,
            "n_candidates_evaluated": len(cv_results),
        },
        "environment": environment_info(),
        "source_sha256_at_freeze": source_hashes(),
    }
    write_json(config.METADATA_PATH, metadata)

    no_skill = {"roc_auc": 0.5, "pr_auc": float(y_test.mean()), "ks": 0.0}
    metrics = {
        "positive_class": "RainTomorrow = Yes",
        "metric_definitions": {
            "roc_auc": "sklearn.metrics.roc_auc_score",
            "pr_auc": "sklearn.metrics.average_precision_score (Average Precision)",
            "ks": "max(TPR - FPR) na curva ROC (KS de duas amostras entre scores de positivos e negativos)",
            "ci": f"bootstrap percentil {int(config.CI_LEVEL * 100)}%, {config.N_BOOTSTRAP} reamostragens, seed {config.RANDOM_STATE}",
        },
        "baseline": {"params": baseline_params, "test": baseline_test},
        "selected_params": params,
        "cv_selected": {k: float(selected_row[k]) for k in
                        ["roc_auc_mean", "roc_auc_se", "pr_auc_mean", "pr_auc_se", "ks_mean", "ks_se"]},
        "train": train_metrics,
        "test": test_metrics,
        "references_test": {"no_skill": no_skill, "persistence_rain_today": persistence_test},
        "oot": None,
        "oot_status": "não avaliado: modelo congelado, OOT ainda fechado",
    }
    write_json(config.METRICS_PATH, metrics)
    print(f"model frozen: {config.MODEL_PATH.relative_to(ROOT)} sha256={model_sha}")
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
