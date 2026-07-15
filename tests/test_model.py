"""
test_model.py
--------------
Basic tests for the training pipeline, inference helper, and the Flask API.

Run (from project root):
    pytest -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))

from predict import predict_risk  # noqa: E402
from preprocessing import FEATURE_COLUMNS, validate_ranges  # noqa: E402

SAMPLE_HIGH_RISK = {
    "Pregnancies": 6,
    "Glucose": 190,
    "BloodPressure": 90,
    "SkinThickness": 40,
    "Insulin": 200,
    "BMI": 40.0,
    "DiabetesPedigreeFunction": 1.5,
    "Age": 55,
}

SAMPLE_LOW_RISK = {
    "Pregnancies": 0,
    "Glucose": 85,
    "BloodPressure": 65,
    "SkinThickness": 18,
    "Insulin": 60,
    "BMI": 21.0,
    "DiabetesPedigreeFunction": 0.15,
    "Age": 22,
}


def test_model_artifact_exists():
    assert (ROOT / "models" / "model.pkl").exists(), (
        "models/model.pkl is missing. Run `python ml/train.py` first."
    )


def test_predict_returns_expected_shape():
    result = predict_risk(SAMPLE_LOW_RISK)
    assert set(result.keys()) == {"probability", "risk_label", "model_name"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_label"] in {"Low Risk", "Moderate Risk", "High Risk"}


def test_predict_directionality():
    """A clearly high-risk profile should score higher than a low-risk one."""
    high = predict_risk(SAMPLE_HIGH_RISK)
    low = predict_risk(SAMPLE_LOW_RISK)
    assert high["probability"] > low["probability"]


def test_validate_ranges_flags_out_of_bounds():
    bad_input = dict(SAMPLE_LOW_RISK)
    bad_input["Glucose"] = 500  # out of the allowed 40-300 range
    errors = validate_ranges(bad_input)
    assert any("Glucose" in e for e in errors)


def test_validate_ranges_flags_missing_field():
    incomplete = {k: v for k, v in SAMPLE_LOW_RISK.items() if k != "BMI"}
    errors = validate_ranges(incomplete)
    assert any("BMI" in e for e in errors)


def test_feature_columns_match_model_input():
    result = predict_risk(SAMPLE_LOW_RISK)
    assert result is not None
    assert len(FEATURE_COLUMNS) == 8


@pytest.fixture
def client():
    from app import app as flask_app

    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Diabetes Risk Assessment" in resp.data


def test_predict_endpoint_success(client):
    resp = client.post("/api/predict", json=SAMPLE_HIGH_RISK)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "probability" in data


def test_predict_endpoint_validation_error(client):
    bad = dict(SAMPLE_HIGH_RISK)
    bad["Age"] = 999
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert "details" in resp.get_json()
