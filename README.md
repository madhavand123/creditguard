# 🛡️ CreditGuard — AI Credit Risk Scoring API

A machine learning system that scores a loan applicant's probability of default in real time, explains the score using SHAP, and serves it through a REST API and an interactive dashboard.

Built with **XGBoost**, **SHAP**, **FastAPI**, **Streamlit**, and **Docker**.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [Model Performance](#-model-performance)
- [Design Decisions](#-design-decisions)
- [Limitations](#-limitations)

---

## Overview

A lender deciding whether to approve a loan needs an estimate of how likely the applicant is to default before saying yes.

CreditGuard takes 12 facts about a borrower (income, credit score, existing debt, etc.) and returns:
1. **A probability of default** — a number between 0 and 1.
2. **A risk grade (A–E)** — a human-readable bucket a loan officer can act on.
3. **The top 5 factors** driving that score, via SHAP.

The dataset is synthetically generated rather than pulled from a real lender (see [Limitations](#-limitations)), so the project can be published and demoed without any privacy concerns, while the statistical relationships in the data (higher credit score → lower risk, more late payments → higher risk, etc.) are modeled to reflect real credit behavior.

For a detailed, section-by-section walkthrough of the codebase, see `10_DAY_GUIDE.md`.

---

## 🏗️ Architecture

```
creditguard/
├── data/
│   ├── generate.py        ← Synthetic credit dataset generator
│   └── credit_data.csv    ← Generated output (20,000 rows)
├── model/
│   ├── train.py           ← Trains XGBoost + computes SHAP importances
│   ├── model.joblib        ← Saved sklearn Pipeline (preprocessing + model)
│   ├── metrics.json        ← ROC-AUC, precision, recall, feature importance
│   └── feature_names.json  ← Ordered feature list (numeric vs categorical)
├── api/
│   └── main.py             ← FastAPI REST endpoints
├── dashboard/
│   └── app.py               ← Streamlit UI (score a borrower + view model stats)
├── risk_grading.py          ← Shared A–E grade logic (imported by both api/ and dashboard/)
├── Dockerfile
└── requirements.txt
```

`risk_grading.py` holds the probability → letter-grade logic in one place, imported by both `api/main.py` and `dashboard/app.py`, so the API and dashboard can't disagree on a borrower's grade.

**Request flow:**
```
Client (curl / Streamlit / browser)
          ↓  HTTP POST /predict
    FastAPI validates input with Pydantic (rejects out-of-range values, e.g. credit_score > 850)
          ↓
    sklearn Pipeline:
      ├── StandardScaler   (numeric features → mean 0, std 1)
      └── OneHotEncoder    (categorical features → binary columns)
          ↓
    XGBoost model → probability of default
          ↓
    SHAP TreeExplainer → per-feature contribution to this specific prediction
          ↓
    risk_grading.assign_grade() → letter grade (A–E)
          ↓
    JSON response → client
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python model/train.py
```
Regenerates the synthetic dataset, trains XGBoost, computes SHAP feature importances, and saves everything to `model/`. Takes ~15–30 seconds.

### 3. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
Swagger docs: http://localhost:8000/docs

### 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Docker
```bash
docker build -t creditguard .
docker run -p 8000:8000 creditguard
```
The model is trained during the image build, so every container started from the image ships with a freshly trained model.

### 6. Verify
```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
```

---

## 📡 API Endpoints

### `POST /predict`
Score a single loan applicant.

**Request:**
```json
{
  "age": 32,
  "annual_income": 65000,
  "credit_score": 710,
  "employment_years": 4.5,
  "num_late_payments": 1,
  "utilization_rate": 0.28,
  "num_credit_lines": 6,
  "loan_amount": 12000,
  "loan_term_months": 36,
  "debt_to_income": 0.22,
  "home_ownership": "RENT",
  "loan_purpose": "DEBT_CONSOLIDATION"
}
```

**Response:**
```json
{
  "probability_of_default": 0.6106,
  "risk_grade": "E",
  "risk_label": "Very High Risk",
  "risk_color": "#ef4444",
  "shap_top_factors": [
    {"feature": "debt_to_income", "shap_value": 0.4316},
    {"feature": "home_ownership_RENT", "shap_value": 0.2592},
    {"feature": "credit_score", "shap_value": -0.244},
    {"feature": "num_late_payments", "shap_value": -0.1061},
    {"feature": "loan_purpose_BUSINESS", "shap_value": -0.0977}
  ],
  "inference_ms": 12.16
}
```

### `POST /predict/batch`
Score up to 100 applicants in one call. Returns a JSON array of the same response shape as `/predict`.

### `GET /model/info`
Returns the algorithm used, training set size, and evaluation metrics.

### `GET /health`
Liveness check — confirms the model loaded and the service is up.

---

## 📊 Model Performance

Metrics from a 20,000-row synthetic dataset, 80/20 train/test split (16,000 train / 4,000 test), stratified by default status:

| Metric    | Value  |
|-----------|--------|
| ROC-AUC   | 0.745  |
| Accuracy  | 0.718  |
| Precision | 0.453  |
| Recall    | 0.593  |
| F1 Score  | 0.514  |

Re-running `python model/train.py` can shift these by a point or two — a known characteristic of XGBoost's multi-threaded histogram tree building, not a bug specific to this project.

---

## 🧠 Design Decisions

| Component    | Why it's here                                                            |
|--------------|---------------------------------------------------------------------------|
| **XGBoost**  | Industry standard for structured/tabular data. Handles non-linear feature interactions (e.g. debt-to-income mattering more when combined with a short loan term) that plain logistic regression would miss. |
| **SHAP**     | Turns a black-box score into "this specific factor pushed the decision this way, by this much" — the standard financial institutions use to satisfy regulatory explainability requirements. |
| **sklearn Pipeline** | Bundles preprocessing (scaling, encoding) and the model into one object, so the scaler's mean/std is fit only on training data — no data leakage from the test set. |
| **FastAPI**  | Async, auto-generates Swagger docs at `/docs`, and Pydantic validates every request before it reaches the model. |
| **Streamlit**| A fast way to build an internal-tool-style UI without writing HTML/JS. |
| **Docker**   | The model is trained and the app runs identically on any machine. |
| **`scale_pos_weight` in XGBoost** | The dataset's default rate is ~25%, so an untuned model could get 75% accuracy by always predicting "no default." This parameter forces the model to weight the minority (default) class appropriately. |

---

## ⚠️ Limitations

- **The data is synthetic.** `data/generate.py` creates borrowers using a hand-built logistic formula, not real loan outcomes. The relationships are directionally realistic, but a production credit model would be trained on actual repayment history and require far more validation (out-of-time testing, population stability checks, fair-lending audits) before informing a real decision.
- **Precision (0.453) is modest.** Of everyone the model flags as high-risk, under half actually default. A production deployment would tune the classification threshold based on the relative cost of a false positive vs a false negative — this project doesn't include that step.
- **SHAP values are in log-odds space, not probability space** (the default for `shap.TreeExplainer` on an XGBoost classifier). They correctly rank which features mattered most for a prediction, but don't literally sum to the displayed probability.
- **No authentication, rate limiting, or logging/monitoring** on the API — out of scope for this project's current form.
