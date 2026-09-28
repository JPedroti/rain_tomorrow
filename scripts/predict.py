"""Score a new CSV with the frozen pipeline - no notebook, no retraining.

    python scripts/predict.py --input data/new_weather.csv --output predictions.csv

The input needs the raw columns of weatherAUS.csv used by the model (Date as YYYY-MM-DD, Location
and the weather predictors; values may be missing). Extra columns - including RainTomorrow - are
ignored. The output keeps the input row order and has one probability per row:

    Date,Location,rain_probability

No threshold is applied: rain_probability is P(RainTomorrow = Yes) estimated by the model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.artifacts import load_frozen_model  # noqa: E402


def unseen_categories(model, frame: pd.DataFrame) -> dict[str, list]:
    """Categories absent from TRAIN; the one-hot encoder maps them to all-zero columns."""
    _, cat, columns = next(t for t in model.named_steps["preprocess"].transformers_ if t[0] == "cat")
    known = cat.named_steps["onehot"].categories_
    out = {}
    for column, categories in zip(columns, known):
        values = set(frame[column].dropna().astype(str).unique())
        new = sorted(values - set(map(str, categories)))
        if new:
            out[column] = new
    return out


def predict(frame: pd.DataFrame, model, metadata: dict) -> pd.DataFrame:
    required = metadata["features"]["input_columns"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    probabilities = model.predict_proba(frame)[:, 1]
    return pd.DataFrame({
        config.DATE_COL: frame[config.DATE_COL].to_numpy(),
        config.LOCATION_COL: frame[config.LOCATION_COL].to_numpy(),
        "rain_probability": probabilities,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="CSV with the raw weather columns")
    parser.add_argument("--output", required=True, type=Path, help="where to write the predictions CSV")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH, help="frozen pipeline (joblib)")
    parser.add_argument("--metadata", type=Path, default=config.METADATA_PATH, help="metadata.json with the model hash")
    args = parser.parse_args()

    model, metadata = load_frozen_model(args.model, args.metadata)
    # Date is kept as text so the output echoes the input exactly.
    frame = pd.read_csv(args.input, dtype={config.DATE_COL: str})
    for column, values in unseen_categories(model, frame).items():
        print(f"WARNING: {column} has categories not seen in training (encoded as all-zero): {values}")

    predictions = predict(frame, model, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, float_format="%.6f")
    print(f"{len(predictions)} rows scored with model sha256={metadata['model_sha256'][:12]}... -> {args.output}")


if __name__ == "__main__":
    main()
