"""Evaluate models/model.joblib on data/eval.csv.

Run with:  python src/eval.py

Model agnostic: imports build_model from src/train.py and reads the model type,
params, feature columns and label column from models/training_config.json
(written by src/train.py). Weights are restored from models/model.joblib.

Prints accuracy, a per-class classification report, and the confusion matrix,
and writes the same results to models/metrics.json.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from train import build_model

# Resolve repo root from this file's location (src/eval.py -> repo root).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

EVAL_PATH = os.path.join(DATA_DIR, "eval.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
CONFIG_PATH = os.path.join(MODELS_DIR, "training_config.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def load_config(path=CONFIG_PATH):
    """Read the training config that describes the model."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    config = load_config()
    feature_columns = config["feature_columns"]
    label_column = config["label_column"]

    # Rebuild the estimator from the recorded type/params, then load weights.
    build_model(config.get("params"))
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(EVAL_PATH)

    y_true = df[label_column]
    y_pred = model.predict(df[feature_columns])

    labels = sorted(y_true.unique())
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "model_type": config["model_type"],
        "eval_path": os.path.relpath(EVAL_PATH, REPO_ROOT),
        "n_eval_rows": len(df),
        "accuracy": accuracy,
        "labels": labels,
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    print(f"model:    {MODEL_PATH} ({config['model_type']})")
    print(f"eval:     {EVAL_PATH} ({len(df)} rows)")
    print(f"accuracy: {accuracy:.4f}")
    print("report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0).rstrip())
    print("confusion matrix:")
    print(matrix)
    print(f"metrics:  {METRICS_PATH}")


if __name__ == "__main__":
    main()
