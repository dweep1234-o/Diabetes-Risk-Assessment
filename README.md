# 🩺 Diabetes Risk Assessment

A full-stack machine learning project that estimates a patient's diabetes
risk from 8 standard clinical inputs. Includes data generation, model
training, evaluation, and a browser-based UI where a patient (or clinician)
can enter details and get an instant risk score.

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## ✨ Features

- **ML pipeline** — StandardScaler + Logistic Regression / Random Forest, model selection by ROC-AUC
- **Patient-facing UI** — clean web form with a live risk gauge, built with Flask + vanilla HTML/CSS/JS (no build step)
- **REST API** — `POST /api/predict` for programmatic access
- **Input validation** — server-side range checks with clear error messages
- **Tests** — pytest suite covering the model, preprocessing, and API
- **CI** — GitHub Actions workflow that regenerates data, retrains, and runs tests on every push
- **Reproducible data** — synthetic dataset generator so the project runs out of the box with no external downloads

## 📁 Project structure

```
diabetes-risk-assessment/
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions: install → generate data → train → test
├── data/
│   ├── raw/
│   │   └── diabetes.csv      # synthetic dataset (generated)
│   └── README.md             # data dictionary
├── ml/
│   ├── generate_data.py      # creates the synthetic dataset
│   ├── preprocessing.py      # shared feature list + input validation
│   ├── train.py               # trains + evaluates + saves the model
│   └── predict.py             # inference helper used by the backend
├── models/
│   ├── model.pkl              # trained pipeline (scaler + classifier)
│   └── metrics.json           # evaluation metrics from the last training run
├── backend/
│   ├── app.py                 # Flask app: serves UI + /api/predict
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js
├── tests/
│   └── test_model.py          # pytest suite
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Clone / open the project folder, then create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (already included, but this regenerates it)
python ml/generate_data.py

# 4. Train the model
python ml/train.py

# 5. Run the app
python backend/app.py
```

Then open **http://127.0.0.1:5000** in your browser, fill in the patient
details, and click **Assess risk**.

### Run the tests

```bash
pytest -v
```

## 🧠 How it works

1. **`ml/generate_data.py`** creates a synthetic dataset shaped like the
   classic 8-feature diabetes risk profile (glucose, BMI, age, blood
   pressure, insulin, pregnancies, skin thickness, and a diabetes pedigree
   score), with a feature→outcome relationship modeled on known clinical
   patterns.
2. **`ml/train.py`** splits the data, trains both a Logistic Regression and
   a Random Forest inside a `StandardScaler` pipeline, evaluates each on a
   held-out test set, and saves whichever scores higher on ROC-AUC to
   `models/model.pkl`.
3. **`backend/app.py`** loads that pipeline once at startup and exposes:
   - `GET /` — the patient-facing form
   - `POST /api/predict` — accepts the 8 feature values as JSON, returns a probability, a risk category (`Low` / `Moderate` / `High`), and the model used
   - `GET /api/health` — health check
4. **`backend/static/js/script.js`** submits the form via `fetch`, then
   animates the result into the on-page gauge.

### Latest evaluation (from `models/metrics.json`)

| Model               | Accuracy | F1   | ROC-AUC |
|----------------------|----------|------|---------|
| Logistic Regression ✅ | 0.72     | 0.67 | **0.77** |
| Random Forest        | 0.70     | 0.64 | 0.76    |

*(Regenerate by re-running `python ml/train.py`; numbers will vary slightly with the random seed.)*

## 🔌 API example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Pregnancies": 2,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
      }'
```

```json
{
  "probability": 0.7637,
  "risk_label": "High Risk",
  "model_name": "logistic_regression"
}
```

## 🔁 Using real clinical data

This project ships with synthetic data so it runs immediately without any
external downloads. To use a real dataset instead (e.g. the public Pima
Indians Diabetes Dataset):

1. Replace `data/raw/diabetes.csv` with a CSV that has the same 9 columns
   (see [`data/README.md`](data/README.md)).
2. Re-run `python ml/train.py`.

## ⚠️ Disclaimer

This tool is an **educational demo**. It estimates statistical risk from a
model trained on synthetic clinical-pattern data and **must not** be used
for real medical diagnosis or treatment decisions. Always consult a




## INTERN ID CITS5351
qualified healthcare provider.

## 📄 License

MIT — see [LICENSE](LICENSE).
