"""
predict.py
----------
Loads the trained pipeline and exposes a simple predict() function that
takes a dict of patient inputs and returns risk probability + label.
"""

from pathlib import Path

import joblib
import pandas as pd

from preprocessing import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run `python ml/train.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_risk(patient: dict) -> dict:
    """
    patient: dict with keys matching FEATURE_COLUMNS
    returns: {"probability": float, "risk_label": str, "model_name": str}
    """
    bundle = _load()
    pipe = bundle["pipeline"]

    row = pd.DataFrame([[patient[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    proba = float(pipe.predict_proba(row)[0, 1])

    if proba < 0.33:
        label = "Low Risk"
    elif proba < 0.66:
        label = "Moderate Risk"
    else:
        label = "High Risk"

    return {
        "probability": round(proba, 4),
        "risk_label": label,
        "model_name": bundle.get("model_name", "unknown"),
    }


if __name__ == "__main__":
    sample = {
        "Pregnancies": 2,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }
    print(predict_risk(sample))
