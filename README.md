# 🛡️ CreditGuard — AI Credit Risk Scoring API

A production-style machine learning system that scores loan applicants' **probability of default** in real time.
Built with XGBoost, SHAP explainability, FastAPI, Streamlit, and Docker.

---

## 🏗️ Architecture

```
creditguard/
├── data/
│   └── generate.py        ← Synthetic credit dataset (10k–20k rows)
├── model/
│   ├── train.py           ← XGBoost training + SHAP analysis
│   ├── model.joblib       ← Saved pipeline (preprocessor + model)
│   └── metrics.json       ← ROC-AUC, precision, recall, feature importance
├── api/
│   └── main.py            ← FastAPI REST endpoints
├── dashboard/
│   └── app.py             ← Streamlit UI (score borrower + model stats)
├── Dockerfile
└── requirements.txt
```

**Event flow:**
```
Raw borrower data → Preprocessing pipeline → XGBoost inference
                                                     ↓
                  SHAP explainer ← Fill score ← Probability of default
                        ↓
              Risk grade (A–E) + top 5 driving factors → API response
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
This generates synthetic data, trains XGBoost, and saves `model/model.joblib`.

### 3. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
Swagger docs: http://localhost:8000/docs

### 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Docker (production)
```bash
docker build -t creditguard .
docker run -p 8000:8000 creditguard
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
  "probability_of_default": 0.1842,
  "risk_grade": "B",
  "risk_label": "Low Risk",
  "risk_color": "#84cc16",
  "shap_top_factors": [
    {"feature": "credit_score", "shap_value": -0.312},
    {"feature": "debt_to_income", "shap_value": 0.187},
    {"feature": "num_late_payments", "shap_value": 0.091}
  ],
  "inference_ms": 12.3
}
```

### `POST /predict/batch`
Score up to 100 applicants in one call.

### `GET /model/info`
Returns ROC-AUC, top features, training metadata.

---

## 📊 Model Performance

| Metric    | Value |
|-----------|-------|
| ROC-AUC   | 0.747 |
| Accuracy  | 0.724 |
| Precision | 0.463 |
| Recall    | 0.594 |
| F1 Score  | 0.520 |

---

## 🧠 Why This Stack?

| Component    | Why it matters                                                            |
|--------------|---------------------------------------------------------------------------|
| **XGBoost**  | Industry standard for tabular credit data. Used by banks globally.        |
| **SHAP**     | Explains *why* a score is high — critical for regulatory compliance (ECOA, GDPR). |
| **FastAPI**  | Async, auto-documented, Pydantic validation — production-grade API design.|
| **Streamlit**| Rapid internal tool UI — exactly how quant teams build dashboards.       |
| **Docker**   | Reproducible deployment — run anywhere, same result.                      |

---

## 💡 Talking Points for Goldman Sachs Interview

- **Scalable API design**: Pydantic validates every request before it hits the model. Batch endpoint handles 100 borrowers per call.
- **Explainability**: SHAP values tell compliance teams *why* someone was denied — this is legally required under ECOA.
- **Risk grading**: A–E grade system mirrors how real lenders (GS, JPMorgan) segment credit risk.
- **Feature engineering**: `debt_to_income` is a derived feature computed from raw inputs — shows financial domain understanding.
- **Imbalanced data**: Used `scale_pos_weight` in XGBoost to handle the natural class imbalance in credit default data.
