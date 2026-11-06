"""Train a classifier on data/train.csv.

Run with:  python src/train.py
      or:  python -m src.train

Fits a scikit-learn DecisionTreeClassifier (see MODEL_TYPE / DEFAULT_PARAMS)
on the training split and writes two artifacts:

  models/model.joblib            fitted estimator ("weights")
  models/training_config.json    model type, columns and params

eval.py imports build_model from this module, so the model kind lives in one
place.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Resolve repo root from this file's location (src/train.py -> repo root).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
CONFIG_PATH = os.path.join(MODELS_DIR, "training_config.json")

LABEL_COLUMN = "species"

DEFAULT_PARAMS = {"random_state": 42, "max_depth": 10}


def feature_columns(df):
    """Return every column except the label, preserving file order."""
    return [column for column in df.columns if column != LABEL_COLUMN]


def build_model(params=None):
    """Instantiate an estimator by name. eval.py reuses this to stay model agnostic."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    return DecisionTreeClassifier(**params)


def main():
    df = pd.read_csv(TRAIN_PATH)
    features = feature_columns(df)

    model = build_model(DEFAULT_PARAMS)
    model.fit(df[features], df[LABEL_COLUMN])

    y_train_pred = model.predict(df[features])
    accuracy = accuracy_score(y_pred=y_train_pred, y_true=df[LABEL_COLUMN])
    f1_macro = f1_score(y_pred=y_train_pred, y_true=df[LABEL_COLUMN], average="macro")
    config = {
        "model_type": model.__class__.__name__,
        "params": DEFAULT_PARAMS,
        "feature_columns": features,
        "label_column": LABEL_COLUMN,
        "train_path": os.path.relpath(TRAIN_PATH, REPO_ROOT),
        "n_train_rows": len(df),
        "training_metrics": {"accuracy": int(accuracy), "f1_macro": int(f1_macro)}
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"train:  {TRAIN_PATH} ({len(df)} rows)")
    print(f"model:  {MODEL_PATH} ({config["model_type"]})")
    print(f"config: {CONFIG_PATH}")
    print(f"Training Accuracy: {accuracy:.2f}")
    print(f'Training F1 Macro: {f1_macro:.2f}')


if __name__ == "__main__":
    main()
