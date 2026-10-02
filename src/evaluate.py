"""Evaluate a saved demo model on the fixed, untouched 20% holdout split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from train import DEFAULT_DATA, ROOT, evaluate, load_data
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic_regression", "random_forest"], default="random_forest")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    X, y, _ = load_data(DEFAULT_DATA)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = joblib.load(ROOT / "models" / f"{args.model}.joblib")
    metrics, _ = evaluate(args.model, model, X_test, y_test, args.threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
