"""
train.py
--------
Trains the diabetes risk classifier and saves the fitted pipeline
(scaler + model) to models/model.pkl, along with a metrics.json report.

Run:
    python ml/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "diabetes.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python ml/generate_data.py` first."
        )
    return pd.read_csv(DATA_PATH)


def build_candidates():
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def main():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_name, best_pipe, best_auc = None, None, -1
    results = {}

    for name, pipe in build_candidates().items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        results[name] = {"accuracy": acc, "f1_score": f1, "roc_auc": auc}
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=["No Diabetes", "Diabetes"]))
        print(f"ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_name, best_pipe, best_auc = name, pipe, auc

    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "model_name": best_name, "features": FEATURE_COLUMNS}, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({"selected_model": best_name, "results": results}, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
