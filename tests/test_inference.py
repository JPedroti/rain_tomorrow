"""predict.py contract, loading in a fresh process and the frozen-hash guard."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import requires_data, requires_model
from src import config
from src.artifacts import load_frozen_model

PREDICT = config.PROJECT_ROOT / "scripts" / "predict.py"
pytestmark = [requires_data, requires_model]


def _run_predict(input_path, output_path):
    return subprocess.run(
        [sys.executable, str(PREDICT), "--input", str(input_path), "--output", str(output_path)],
        capture_output=True, text=True, cwd=config.PROJECT_ROOT,
    )


@pytest.fixture(scope="module")
def sample_input(dev_split, tmp_path_factory):
    """Rows from the historical TEST set, shuffled, with the target column still present."""
    sample = dev_split["test"].sample(300, random_state=0).copy()
    sample["Date"] = sample["Date"].dt.strftime("%Y-%m-%d")
    sample = sample[list(reversed(sample.columns))]  # column order must not matter
    path = tmp_path_factory.mktemp("predict") / "new_weather.csv"
    sample.to_csv(path, index=False)
    return path, sample


def test_predict_script_contract(sample_input, tmp_path):
    input_path, sample = sample_input
    out = tmp_path / "predictions.csv"
    result = _run_predict(input_path, out)
    assert result.returncode == 0, result.stderr
    pred = pd.read_csv(out, dtype={"Date": str})
    assert pred.columns.tolist() == ["Date", "Location", "rain_probability"]
    assert len(pred) == len(sample)
    assert pred["Date"].tolist() == sample["Date"].tolist()          # row order preserved
    assert pred["Location"].tolist() == sample["Location"].tolist()
    assert pred["rain_probability"].between(0, 1).all()


def test_script_output_equals_in_process_model(sample_input, tmp_path):
    input_path, _ = sample_input
    out = tmp_path / "predictions.csv"
    assert _run_predict(input_path, out).returncode == 0
    model, _ = load_frozen_model()
    frame = pd.read_csv(input_path, dtype={"Date": str})
    expected = model.predict_proba(frame)[:, 1]
    np.testing.assert_allclose(pd.read_csv(out)["rain_probability"], expected, atol=1e-6)


def test_missing_required_column_fails_clearly(sample_input, tmp_path):
    _, sample = sample_input
    broken = tmp_path / "broken.csv"
    sample.drop(columns=["Humidity3pm"]).to_csv(broken, index=False)
    result = _run_predict(broken, tmp_path / "out.csv")
    assert result.returncode != 0
    assert "Humidity3pm" in result.stderr


def test_fresh_process_reproduces_frozen_test_metrics():
    code = (
        "from src.artifacts import load_frozen_model, read_json; from src.data import load_train_test, get_xy;"
        "from src.metrics import discrimination_metrics; from src import config; import json;"
        "m,_=load_frozen_model(); X,y=get_xy(load_train_test()['test']);"
        "print(json.dumps(discrimination_metrics(y, m.predict_proba(X)[:,1])))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    fresh = json.loads(result.stdout.strip().splitlines()[-1])
    recorded = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))["test"]
    for metric in ("roc_auc", "pr_auc", "ks"):
        assert fresh[metric] == pytest.approx(recorded[metric], abs=1e-12)


def test_tampered_model_is_rejected(tmp_path):
    model_copy = tmp_path / "model.joblib"
    shutil.copy(config.MODEL_PATH, model_copy)
    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    metadata["model_sha256"] = "0" * 64
    fake_meta = tmp_path / "metadata.json"
    fake_meta.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        load_frozen_model(model_copy, fake_meta)
