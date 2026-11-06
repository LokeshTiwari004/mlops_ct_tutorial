"""Split data/iris.csv into stratified train/eval CSVs.

Run with:  python src/split.py
      or:  python -m src.split

Writes data/train.csv and data/eval.csv using scikit-learn's
train_test_split with stratify=y, so the species balance is preserved in both
outputs. The split is reproducible via a fixed random seed.
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve repo root from this file's location (src/split.py -> repo root).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
INPUT_PATH = os.path.join(DATA_DIR, "iris.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
EVAL_PATH = os.path.join(DATA_DIR, "eval.csv")

LABEL_COLUMN = "species"

DEFAULT_EVAL_SIZE = 0.2
DEFAULT_SEED = 42


def split(iris, eval_size=DEFAULT_EVAL_SIZE, seed=DEFAULT_SEED):
    """Return (train_df, eval_df) as a stratified shuffle split of iris."""
    train_df, eval_df = train_test_split(
        iris,
        test_size=eval_size,
        stratify=iris[LABEL_COLUMN],
        random_state=seed,
        shuffle=True,
    )
    return train_df, eval_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-size",
        type=float,
        default=DEFAULT_EVAL_SIZE,
        help="fraction held out for eval (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="random seed for reproducible splits (default: %(default)s)",
    )
    args = parser.parse_args()

    iris = pd.read_csv(INPUT_PATH)
    train_df, eval_df = split(iris, args.eval_size, args.seed)

    train_df.to_csv(TRAIN_PATH, index=False)
    eval_df.to_csv(EVAL_PATH, index=False)

    print(f"input: {INPUT_PATH} ({len(iris)} rows)")
    print(f"train: {TRAIN_PATH} ({len(train_df)} rows)")
    print(f"eval:  {EVAL_PATH} ({len(eval_df)} rows)")


if __name__ == "__main__":
    main()
