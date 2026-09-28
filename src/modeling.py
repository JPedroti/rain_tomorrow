"""LogisticRegression pipeline and the pre-registered model-selection protocol (TRAIN-only CV)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.data import assert_no_oot
from src.metrics import discrimination_metrics
from src.preprocessing import build_preprocessor

METRICS = ("roc_auc", "pr_auc", "ks")


def solver_for(l1_ratio: float) -> str:
    """lbfgs for the L2 penalty; saga supports L1 (and does not penalise the intercept, unlike liblinear)."""
    return "lbfgs" if l1_ratio == 0.0 else "saga"


def penalty_name(l1_ratio: float) -> str:
    return {0.0: "l2", 1.0: "l1"}.get(l1_ratio, f"elasticnet({l1_ratio})")


def build_model(C: float, l1_ratio: float, class_weight) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver=solver_for(l1_ratio),
        class_weight=class_weight,
        max_iter=config.MAX_ITER,
        random_state=config.RANDOM_STATE,
    )


def build_pipeline(
    feature_set: str = config.FINAL_FEATURE_SET,
    preprocessing: str = config.BASELINE_PREPROCESSING,
    C: float = 1.0,
    l1_ratio: float = 0.0,
    class_weight=None,
) -> Pipeline:
    spec = config.FEATURE_SETS[feature_set]
    variant = config.PREPROCESSING_VARIANTS[preprocessing]
    return Pipeline([
        ("preprocess", build_preprocessor(spec["numeric"], spec["categorical"], **variant)),
        ("model", build_model(C, l1_ratio, class_weight)),
    ])


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def _fit_and_score(params: dict, X: pd.DataFrame, y: np.ndarray, train_idx, val_idx) -> dict:
    """Fit the whole pipeline on the training fold only and score the held-out fold."""
    start = time.perf_counter()
    pipe = build_pipeline(**params).fit(X.iloc[train_idx], y[train_idx])
    fit_time = time.perf_counter() - start
    scores = discrimination_metrics(y[val_idx], pipe.predict_proba(X.iloc[val_idx])[:, 1])
    return {**{m: scores[m] for m in METRICS}, "fit_time": fit_time}


def evaluate_candidates(candidates: list[dict], X: pd.DataFrame, y: np.ndarray, n_jobs: int = -1) -> pd.DataFrame:
    """Cross-validate every candidate on the same stratified folds (parallel over candidate x fold).

    `candidates` order is the preference order used by the one-standard-error rule.
    """
    assert_no_oot(X)
    folds = list(cv_splitter().split(X, y))
    tasks = [(c, f) for c in range(len(candidates)) for f in range(len(folds))]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_score)(candidates[c], X, y, *folds[f]) for c, f in tasks
    )
    by_candidate: dict[int, list[dict]] = {}
    for (c, _), res in zip(tasks, results):
        by_candidate.setdefault(c, []).append(res)

    rows = []
    for order, params in enumerate(candidates):
        fold_results = by_candidate[order]
        row = {"order": order, **params, "penalty": penalty_name(params.get("l1_ratio", 0.0)),
               "fit_time_mean_s": float(np.mean([r["fit_time"] for r in fold_results]))}
        for metric in METRICS:
            fold = np.array([r[metric] for r in fold_results])
            row[f"{metric}_mean"] = float(fold.mean())
            row[f"{metric}_std"] = float(fold.std(ddof=1))
            row[f"{metric}_se"] = float(fold.std(ddof=1) / np.sqrt(len(fold)))
            for i, value in enumerate(fold):
                row[f"{metric}_fold{i}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def select_one_se(results: pd.DataFrame, metric: str = config.SELECTION_METRIC) -> int:
    """One-standard-error rule: first candidate (in preference order) within best_mean - best_SE."""
    best = results.loc[results[f"{metric}_mean"].idxmax()]
    threshold = best[f"{metric}_mean"] - best[f"{metric}_se"]
    eligible = results.loc[results[f"{metric}_mean"] >= threshold].sort_values("order")
    return int(eligible.index[0])


def preprocessing_candidates(feature_set: str) -> list[dict]:
    return [
        {"feature_set": feature_set, "preprocessing": name, **config.BASELINE_MODEL_PARAMS}
        for name in config.PREPROCESSING_VARIANTS  # dict order = simplest first
    ]


def hyperparameter_candidates(feature_set: str, preprocessing: str) -> list[dict]:
    candidates = [
        {"feature_set": feature_set, "preprocessing": preprocessing,
         "C": C, "l1_ratio": l1_ratio, "class_weight": class_weight}
        for class_weight in config.CLASS_WEIGHT_GRID
        for C in config.C_GRID
        for l1_ratio in config.L1_RATIO_GRID
    ]
    # Preference order: class_weight=None first, then smaller C, then L2 before L1.
    return sorted(candidates, key=lambda p: (p["class_weight"] is not None, p["C"], p["l1_ratio"]))


def run_model_selection(train: pd.DataFrame, y_train: np.ndarray, feature_set: str, n_jobs: int = -1) -> dict:
    """Stage 1 picks the preprocessing (baseline LR); stage 2 tunes LR on it. TRAIN rows only."""
    assert_no_oot(train)
    stage1 = evaluate_candidates(preprocessing_candidates(feature_set), train, y_train, n_jobs=n_jobs)
    stage1_idx = select_one_se(stage1)
    stage1["stage"] = "1_preprocessing"
    stage1["selected"] = stage1.index == stage1_idx
    preprocessing = stage1.loc[stage1_idx, "preprocessing"]

    stage2 = evaluate_candidates(hyperparameter_candidates(feature_set, preprocessing), train, y_train, n_jobs=n_jobs)
    stage2_idx = select_one_se(stage2)
    stage2["stage"] = "2_hyperparameters"
    stage2["selected"] = stage2.index == stage2_idx
    chosen = stage2.loc[stage2_idx]

    params = {
        "feature_set": feature_set,
        "preprocessing": preprocessing,
        "C": float(chosen["C"]),
        "l1_ratio": float(chosen["l1_ratio"]),
        "class_weight": None if pd.isna(chosen["class_weight"]) else chosen["class_weight"],
    }
    return {"cv_results": pd.concat([stage1, stage2], ignore_index=True), "selected_params": params}


def persistence_score(df: pd.DataFrame) -> np.ndarray:
    """Reference 'model' (weather persistence): 1 if it rained today, 0 if not, 0.5 (middle rank) if unknown."""
    return df["RainToday"].map({"Yes": 1.0, "No": 0.0}).fillna(0.5).to_numpy(dtype=float)


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients on the standardised / one-hot feature space, sorted by absolute value."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coef})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False, ignore_index=True)
