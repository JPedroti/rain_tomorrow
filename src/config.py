"""Project-wide constants: paths, temporal cutoff, feature lists and the pre-registered selection protocol.

Everything that defines the experiment lives here so that scripts, tests and the notebook share one
source of truth. Values were fixed from the brief and from the phase-1 audit of data before 2017.
"""
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# Paths (relative to the project root, resolved at import time)
# --------------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "weatherAUS.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"
SENSITIVITY_PATH = REPORTS_DIR / "sensitivity_analysis.csv"
TEMPORAL_PERFORMANCE_PATH = REPORTS_DIR / "temporal_performance.csv"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.csv"
TARGET_DRIFT_PATH = REPORTS_DIR / "target_drift.csv"

# Kaggle "Rain in Australia" (jsphyg/weather-dataset-rattle-package), Version 2.
DATASET_SHA256 = "573fd715cd69fcacc4df32024d823b450ae3edaae7e8ff2eeb623adbed424014"

# --------------------------------------------------------------------------------------------------
# Columns, target and temporal cutoff
# --------------------------------------------------------------------------------------------------
DATE_COL = "Date"
LOCATION_COL = "Location"
TARGET_COL = "RainTomorrow"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

# Every row dated on/after this day is Out-of-Time and never used for development decisions.
OOT_START = "2017-01-01"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]
CATEGORICAL_FEATURES = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # the 21 original predictors
# Heavy right tail with many zeros; log1p is a candidate transformation evaluated by CV.
SKEWED_FEATURES = ["Rainfall", "Evaporation"]

# Measurement window of each raw predictor (Bureau of Meteorology "Notes to accompany Daily Weather
# Observations" and the Kaggle data card). The target window is 09:00 of day t -> 09:00 of day t+1.
FEATURE_WINDOWS = {
    "MinTemp": "24h até 09:00 de t",
    "Rainfall": "24h até 09:00 de t",
    "RainToday": "24h até 09:00 de t",
    "Evaporation": "24h até 09:00 de t",
    "Temp9am": "instante 09:00 de t",
    "Humidity9am": "instante 09:00 de t",
    "Cloud9am": "instante 09:00 de t",
    "Pressure9am": "instante 09:00 de t",
    "WindDir9am": "média 10 min antes de 09:00 de t",
    "WindSpeed9am": "média 10 min antes de 09:00 de t",
    "Temp3pm": "instante 15:00 de t",
    "Humidity3pm": "instante 15:00 de t",
    "Cloud3pm": "instante 15:00 de t",
    "Pressure3pm": "instante 15:00 de t",
    "WindDir3pm": "média 10 min antes de 15:00 de t",
    "WindSpeed3pm": "média 10 min antes de 15:00 de t",
    "Sunshine": "24h até 00:00 de t+1",
    "WindGustDir": "24h até 00:00 de t+1",
    "WindGustSpeed": "24h até 00:00 de t+1",
    "MaxTemp": "24h de 09:00 de t até 09:00 de t+1",
    "Location": "fixo",
}

# Feature sets. Date always contributes day-of-year sin/cos (never the year).
#   full_without_maxtemp : FINAL MODEL. 20 raw predictors, all closed by the end of day t (00:00 of t+1).
#                          MaxTemp is excluded because its window (09:00 t -> 09:00 t+1) is the target window
#                          and only closes after the prediction moment (decision approved on 2026-09-28).
#   full                 : all 21 raw predictors, including MaxTemp (sensitivity analysis only).
#   morning              : only predictors whose window closes by 09:00 of day t (sensitivity analysis only).
FEATURE_SETS = {
    "full": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    "full_without_maxtemp": {
        "numeric": [f for f in NUMERIC_FEATURES if f != "MaxTemp"],
        "categorical": CATEGORICAL_FEATURES,
    },
    "morning": {
        "numeric": ["MinTemp", "Rainfall", "Evaporation", "WindSpeed9am",
                    "Humidity9am", "Pressure9am", "Cloud9am", "Temp9am"],
        "categorical": ["Location", "WindDir9am", "RainToday"],
    },
}
FINAL_FEATURE_SET = "full_without_maxtemp"

# --------------------------------------------------------------------------------------------------
# Pre-registered model-selection protocol (TRAIN only, stratified K-fold CV)
# --------------------------------------------------------------------------------------------------
# Selection metric: mean CV ROC-AUC. Rule: one-standard-error rule. Among candidates whose mean
# ROC-AUC >= best_mean - best_SE (SE = fold std / sqrt(k), ddof=1), pick the first in the candidate
# order below (simplest / most regularised first).
SELECTION_METRIC = "roc_auc"

# Stage 1: preprocessing variants, listed from simplest to most complex.
PREPROCESSING_VARIANTS = {
    "base": {"missing_indicators": False, "log_transform": False},
    "indicators": {"missing_indicators": True, "log_transform": False},
    "indicators_log": {"missing_indicators": True, "log_transform": True},
}
BASELINE_MODEL_PARAMS = {"C": 1.0, "l1_ratio": 0.0, "class_weight": None}
BASELINE_PREPROCESSING = "base"

# Stage 2: LogisticRegression hyper-parameters on the selected preprocessing.
# scikit-learn 1.9 deprecates `penalty`; l1_ratio=0 is the L2 penalty and l1_ratio=1 the L1 penalty.
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
L1_RATIO_GRID = [0.0, 1.0]
CLASS_WEIGHT_GRID = [None, "balanced"]
# Candidate order inside the 1-SE band: class_weight=None first (keeps predicted probabilities on the
# scale of the observed prevalence), then smaller C (stronger regularisation), then L2 before L1.

MAX_ITER = 5000

# --------------------------------------------------------------------------------------------------
# Evaluation protocol (fixed before the OOT is opened)
# --------------------------------------------------------------------------------------------------
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
# A monthly window is evaluated only with enough volume in both classes; otherwise metrics are NaN.
MIN_WINDOW_ROWS = 200
MIN_WINDOW_PER_CLASS = 30
# Best / worst OOT month are defined by ROC-AUC, the selection metric.
PERIOD_RANKING_METRIC = "roc_auc"
# Drift: decile bins learned on TRAIN (plus a missing bin); heuristic PSI reading cut-offs.
PSI_BINS = 10
PSI_EPS = 1e-4
PSI_REFERENCE_CUTOFFS = (0.10, 0.25)
# Illustrative thresholds for the didactic confusion matrices (no threshold is selected).
ILLUSTRATIVE_THRESHOLDS = [0.3, 0.5, 0.7]
