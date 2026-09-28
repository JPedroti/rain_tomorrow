"""Persistence of the frozen model and of the JSON artifacts.

The model is frozen by recording its SHA-256 in metadata.json at training time. Every consumer
(evaluate.py, predict.py, notebook) loads it through `load_frozen_model`, which refuses a file whose
hash differs from the recorded one.
"""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.data import file_sha256

MODEL_AFFECTING_SOURCES = [
    "src/config.py", "src/data.py", "src/preprocessing.py", "src/modeling.py", "src/metrics.py",
    "scripts/train.py",
]


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model(model, path: Path = config.MODEL_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return file_sha256(path)


def load_frozen_model(model_path: Path = config.MODEL_PATH, metadata_path: Path = config.METADATA_PATH):
    """Load the pipeline after checking its SHA-256 against the frozen metadata."""
    metadata = read_json(metadata_path)
    actual = file_sha256(model_path)
    expected = metadata["model_sha256"]
    if actual != expected:
        raise RuntimeError(f"{model_path} does not match the frozen model (sha256 {actual} != {expected}).")
    return joblib.load(model_path), metadata


def source_hashes() -> dict:
    return {rel: file_sha256(config.PROJECT_ROOT / rel) for rel in MODEL_AFFECTING_SOURCES}


def environment_info() -> dict:
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
