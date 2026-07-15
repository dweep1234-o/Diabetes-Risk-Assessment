"""
generate_data.py
-----------------
Generates a synthetic patient dataset for the Diabetes Risk Assessment project.

NOTE ON DATA:
This project ships with a SYNTHETIC dataset (not real patient records). The
feature distributions and the relationship between features and diabetes risk
are modeled after well-known clinical patterns for the 8 standard risk
factors (as popularized by the Pima Indians Diabetes Dataset), so the model
behaves realistically for demo / learning purposes. For a production system,
replace `data/raw/diabetes.csv` with a real, ethically-sourced clinical
dataset and re-run `ml/train.py`.

Run:
    python ml/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 2000

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "diabetes.csv"


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pregnancies = rng.poisson(lam=2.2, size=n_samples).clip(0, 15)
    age = rng.gamma(shape=6, scale=6, size=n_samples).clip(18, 85).astype(int)
    bmi = rng.normal(loc=30, scale=7, size=n_samples).clip(14, 60).round(1)
    glucose = rng.normal(loc=115, scale=30, size=n_samples).clip(50, 250).round(0)
    blood_pressure = rng.normal(loc=72, scale=13, size=n_samples).clip(30, 130).round(0)
    skin_thickness = rng.normal(loc=23, scale=10, size=n_samples).clip(0, 60).round(0)
    insulin = rng.gamma(shape=2.0, scale=60, size=n_samples).clip(0, 600).round(0)
    dpf = rng.gamma(shape=1.5, scale=0.25, size=n_samples).clip(0.05, 2.5).round(3)

    # Latent risk score built from known clinically-relevant weighted factors,
    # then squashed through a logistic function to produce a probability,
    # from which the binary outcome label is sampled (adds realistic noise).
    z = (
        -10.2
        + 0.038 * glucose
        + 0.085 * bmi
        + 0.021 * age
        + 0.90 * dpf
        + 0.10 * pregnancies
        + 0.004 * insulin
        + 0.012 * blood_pressure
    )
    prob = 1 / (1 + np.exp(-z))
    outcome = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
            "Outcome": outcome,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df["Outcome"].value_counts(normalize=True).rename("class_balance"))
