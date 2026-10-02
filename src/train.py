"""Train reproducible demo models for the synthetic loan-status dataset.

This project is an educational classification example.  It must not be used to
approve, deny, price, or otherwise make real lending decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "loan_data.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
TARGET = "loan_status"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load data, remove ID-like fields, and return X/y."""
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column '{TARGET}', found {list(data.columns)}")
    identifier_columns = [
        column for column in data.columns
        if column != TARGET and (column.lower() == "id" or column.lower().endswith("_id")
        or column.lower().startswith("unnamed"))
    ]
    X = data.drop(columns=[TARGET, *identifier_columns])
    y = data[TARGET].astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError("loan_status must be binary encoded as 0 and 1.")
    return X, y, identifier_columns


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def build_searches(preprocessor: ColumnTransformer) -> dict[str, GridSearchCV]:
    """Create small, transparent searches: all tuning stays inside training CV."""
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=2_000, solver="liblinear", random_state=42),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [200],
                "model__max_depth": [None, 16],
                "model__min_samples_leaf": [1, 5],
            },
        ),
    }
    return {
        name: GridSearchCV(
            Pipeline([("preprocessor", preprocessor), ("model", estimator)]),
            parameters,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            refit=True,
        )
        for name, (estimator, parameters) in candidates.items()
    }


def evaluate(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
             threshold: float) -> tuple[dict[str, float | list[list[int]]], np.ndarray]:
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    metrics = {
        "model": name,
        "threshold": threshold,
        "precision_positive": round(float(precision_score(y_test, prediction, zero_division=0)), 4),
        "recall_positive": round(float(recall_score(y_test, prediction, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
    }
    return metrics, probability


def save_figures(results: dict[str, dict], probabilities: dict[str, np.ndarray],
                 y_test: pd.Series, threshold: float) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, probability in probabilities.items():
        RocCurveDisplay.from_predictions(y_test, probability, name=name.replace("_", " ").title(), ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set_title("ROC curves — untouched test set")
    axes[0].grid(alpha=0.25)
    selected = max(results, key=lambda key: results[key]["roc_auc"])
    ConfusionMatrixDisplay(
        confusion_matrix=np.array(results[selected]["confusion_matrix"]),
        display_labels=["0: not approved", "1: approved"],
    ).plot(ax=axes[1], colorbar=False)
    axes[1].set_title(f"{selected.replace('_', ' ').title()} at threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    X, y, dropped = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = make_preprocessor(X_train)
    searches = build_searches(preprocessor)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    results, probabilities = {}, {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        metrics, probability = evaluate(name, search.best_estimator_, X_test, y_test, args.threshold)
        metrics["cv_roc_auc"] = round(float(search.best_score_), 4)
        metrics["best_parameters"] = search.best_params_
        results[name] = metrics
        probabilities[name] = probability
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")

    payload = {
        "dataset": str(args.data.relative_to(ROOT)),
        "synthetic_data_notice": "Educational demo only; not a real lending decision system.",
        "target": TARGET,
        "dropped_identifier_columns": dropped,
        "features": X.columns.tolist(),
        "split": {"train_rows": len(X_train), "test_rows": len(X_test), "random_state": 42, "stratified": True},
        "cross_validation": "Stratified 5-fold GridSearchCV on training data only, scored by ROC-AUC.",
        "models": results,
    }
    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    save_figures(results, probabilities, y_test, args.threshold)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
