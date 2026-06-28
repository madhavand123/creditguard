"""
main.py — FastAPI Credit Risk Scoring API

Endpoints:
  POST /predict         — score a single borrower
  POST /predict/batch   — score up to 100 borrowers at once
  GET  /health          — service health check
  GET  /model/info      — model metadata and performance metrics

Run with:  uvicorn api.main:app --reload --port 8000
Docs at:   http://localhost:8000/docs   (auto-generated Swagger UI)
"""

import json, pathlib, time
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Load model artifacts ────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent.parent
MODEL    = joblib.load(BASE / "model" / "model.joblib")
METRICS  = json.loads((BASE / "model" / "metrics.json").read_text())
FEATURES = json.loads((BASE / "model" / "feature_names.json").read_text())

EXPLAINER = shap.TreeExplainer(MODEL.named_steps["model"])

RISK_GRADES = {
    (0.00, 0.10): ("A", "Very Low Risk",  "#22c55e"),
    (0.10, 0.20): ("B", "Low Risk",       "#84cc16"),
    (0.20, 0.35): ("C", "Moderate Risk",  "#f59e0b"),
    (0.35, 0.55): ("D", "High Risk",      "#f97316"),
    (0.55, 1.01): ("E", "Very High Risk", "#ef4444"),
}


def assign_grade(prob: float) -> dict:
    for (lo, hi), (grade, label, color) in RISK_GRADES.items():
        if lo <= prob < hi:
            return {"grade": grade, "label": label, "color": color}
    return {"grade": "E", "label": "Very High Risk", "color": "#ef4444"}


# ── Pydantic request/response schemas ───────────────────────────────────────
class BorrowerInput(BaseModel):
    age:               int   = Field(..., ge=18, le=100,   example=32)
    annual_income:     float = Field(..., ge=0,            example=65000)
    credit_score:      int   = Field(..., ge=300, le=850,  example=710)
    employment_years:  float = Field(..., ge=0, le=50,     example=4.5)
    num_late_payments: int   = Field(..., ge=0,            example=1)
    utilization_rate:  float = Field(..., ge=0.0, le=1.0,  example=0.28)
    num_credit_lines:  int   = Field(..., ge=0,            example=6)
    loan_amount:       float = Field(..., ge=0,            example=12000)
    loan_term_months:  int   = Field(..., ge=1,            example=36)
    debt_to_income:    float = Field(..., ge=0.0, le=1.0,  example=0.22)
    home_ownership:    str   = Field(...,                  example="RENT")
    loan_purpose:      str   = Field(...,                  example="DEBT_CONSOLIDATION")

    @field_validator("home_ownership")
    @classmethod
    def validate_home(cls, v):
        valid = {"RENT", "OWN", "MORTGAGE"}
        if v.upper() not in valid:
            raise ValueError(f"home_ownership must be one of {valid}")
        return v.upper()

    @field_validator("loan_purpose")
    @classmethod
    def validate_purpose(cls, v):
        valid = {"DEBT_CONSOLIDATION", "MEDICAL", "HOME_IMPROVEMENT", "EDUCATION", "BUSINESS"}
        if v.upper() not in valid:
            raise ValueError(f"loan_purpose must be one of {valid}")
        return v.upper()


class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_grade:   str
    risk_label:   str
    risk_color:   str
    shap_top_factors: List[dict]   # top 5 features driving this score
    inference_ms: float


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "CreditGuard — AI Credit Risk API",
    description = "XGBoost + SHAP credit default scoring for loan applicants.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _predict_single(borrower: BorrowerInput) -> PredictionResponse:
    start = time.perf_counter()

    row = pd.DataFrame([borrower.model_dump()])
    prob = float(MODEL.predict_proba(row)[0, 1])

    # SHAP explanation for this specific borrower
    transformed = MODEL.named_steps["prep"].transform(row)
    sv           = EXPLAINER.shap_values(transformed)[0]
    feat_names   = FEATURES["all"]

    top_factors = sorted(
        [{"feature": n, "shap_value": round(float(v), 4)}
         for n, v in zip(feat_names, sv)],
        key=lambda x: abs(x["shap_value"]), reverse=True
    )[:5]

    grade_info = assign_grade(prob)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return PredictionResponse(
        probability_of_default = round(prob, 4),
        risk_grade             = grade_info["grade"],
        risk_label             = grade_info["label"],
        risk_color             = grade_info["color"],
        shap_top_factors       = top_factors,
        inference_ms           = elapsed_ms,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "1.0.0"}


@app.get("/model/info")
def model_info():
    return {
        "algorithm":     "XGBoost + StandardScaler + OneHotEncoder",
        "training_rows": 8000,
        "metrics":       {k: v for k, v in METRICS.items()
                          if k not in ("confusion_matrix", "feature_importance")},
        "top_features":  list(METRICS.get("feature_importance", {}).keys())[:6],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(borrower: BorrowerInput):
    """Score a single loan applicant and return SHAP-based explanations."""
    try:
        return _predict_single(borrower)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(borrowers: List[BorrowerInput]):
    """Score up to 100 applicants in one call."""
    if len(borrowers) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100.")
    return [_predict_single(b) for b in borrowers]
