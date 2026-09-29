"""OOT-stage analyses and figures exercised on synthetic data (the real OOT is never needed here)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import make_weather_frame  # noqa: E402
from src import config, plots  # noqa: E402
from src.data import get_xy  # noqa: E402
from src.evaluation import feature_drift_report, monthly_performance, target_drift_report  # noqa: E402
from src.modeling import build_pipeline, coefficient_table  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_run():
    train = make_weather_frame(n=3000, seed=11)
    test = make_weather_frame(n=1000, seed=12)
    oot = make_weather_frame(n=1800, seed=13, start="2017-01-01", end="2017-06-25")
    X_train, y_train = get_xy(train)
    model = build_pipeline(feature_set="full", preprocessing="indicators").fit(X_train, y_train)
    scores = {}
    for name, frame in {"train": train, "test": test, "oot": oot}.items():
        X, y = get_xy(frame)
        scores[name] = (y, model.predict_proba(X)[:, 1])
    return train, test, oot, model, scores


def test_monthly_performance_has_one_row_per_oot_month_with_history(synthetic_run):
    train, test, oot, _, s = synthetic_run
    monthly = monthly_performance(oot, *s["oot"], test, *s["test"])
    assert monthly["period"].tolist() == [f"2017-0{m}" for m in range(1, 7)]
    assert monthly["n"].sum() == len(oot)
    assert {"hist_test_roc_auc", "delta_roc_auc_vs_hist_test", "roc_auc_ci_low"} <= set(monthly.columns)
    assert monthly["hist_test_n"].notna().all()


def test_target_drift_report_rates(synthetic_run):
    train, test, oot, _, _ = synthetic_run
    by_month, rates = target_drift_report(train, test, oot)
    assert rates["oot"] == pytest.approx((oot[config.TARGET_COL] == "Yes").mean())
    assert by_month["oot_rate"].notna().sum() == 6 and by_month["train_rate"].notna().sum() == 12
    assert by_month["test_rate"].notna().sum() == 12
    assert by_month["oot_n"].sum() == oot[config.TARGET_COL].notna().sum()
    assert 0 <= rates["oot_expected_from_train_seasonality"] <= 1


def test_feature_drift_report_covers_both_references(synthetic_run):
    train, _, oot, _, s = synthetic_run
    drift = feature_drift_report(train, oot, s["train"][1], s["oot"][1])
    assert drift["reference"].nunique() == 2
    per_ref = drift.groupby("reference")["feature"].apply(set)
    for features in per_ref:
        assert set(config.NUMERIC_FEATURES) | {"Location", "score do modelo"} == features
    assert drift["psi"].notna().all()
    in_model = drift.set_index("feature")["in_model"].groupby(level=0).first()
    assert not in_model["MaxTemp"] and in_model["Humidity3pm"] and in_model["Location"]


def test_monthly_performance_reports_days_covered(synthetic_run):
    _, test, oot, _, s = synthetic_run
    monthly = monthly_performance(oot, *s["oot"], test, *s["test"])
    assert (monthly["days_covered"] <= 31).all() and monthly["days_covered"].min() >= 1


def test_figures_render(synthetic_run, tmp_path):
    train, test, oot, model, s = synthetic_run
    monthly = monthly_performance(oot, *s["oot"], test, *s["test"])
    by_month, _ = target_drift_report(train, test, oot)
    drift = feature_drift_report(train, oot, s["train"][1], s["oot"][1])
    cv = pd.DataFrame({"stage": "2_hyperparameters", "class_weight": [None, None, "balanced", "balanced"],
                       "penalty": ["l2", "l1", "l2", "l1"], "C": [1.0, 1.0, 1.0, 1.0],
                       "roc_auc_mean": [0.8, 0.79, 0.81, 0.8], "roc_auc_se": 0.01, "selected": [True, False, False, False]})
    ref_cols = {f"hist_test_{m}": m for m in ["roc_auc", "pr_auc", "ks", "prevalence"]}
    figures = [
        plots.plot_roc_pr({"TEST": s["test"], "OOT": s["oot"]}, path=tmp_path / "a.png"),
        plots.plot_score_distributions(*s["oot"], path=tmp_path / "b.png"),
        plots.plot_cv_path(cv, path=tmp_path / "c.png"),
        plots.plot_coefficients(coefficient_table(model), path=tmp_path / "d.png"),
        plots.plot_monthly_performance(monthly, monthly[["month", *ref_cols]].rename(columns=ref_cols), "hist", tmp_path / "e.png"),
        plots.plot_target_rate(by_month, path=tmp_path / "f.png"),
        plots.plot_drift(drift[drift["type"] == "numérica"], path=tmp_path / "g.png"),
    ]
    assert len(list(tmp_path.glob("*.png"))) == len(figures)
    plt.close("all")
