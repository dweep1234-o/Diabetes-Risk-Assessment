"""
preprocessing.py
-----------------
Shared preprocessing logic used by both training (train.py) and inference
(predict.py / backend) so the exact same transformations are applied at
prediction time as at training time.
"""

from dataclasses import dataclass, field
from typing import List

FEATURE_COLUMNS: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN = "Outcome"

# Reasonable clinical bounds used for basic input validation in the UI/API.
FEATURE_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 300),
    "BloodPressure": (20, 180),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (10.0, 70.0),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (1, 120),
}


@dataclass
class PatientInput:
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

    def to_feature_list(self) -> list:
        return [getattr(self, col) for col in FEATURE_COLUMNS]


def validate_ranges(values: dict) -> list:
    """Returns a list of human-readable error strings, empty if all valid."""
    errors = []
    for col, (lo, hi) in FEATURE_RANGES.items():
        if col not in values:
            errors.append(f"Missing field: {col}")
            continue
        v = values[col]
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{col} must be a number.")
            continue
        if not (lo <= v <= hi):
            errors.append(f"{col} should be between {lo} and {hi}.")
    return errors
