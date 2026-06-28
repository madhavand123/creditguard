"""
app.py — Streamlit dashboard for CreditGuard.

Shows:
  1. Borrower input form → live risk score
  2. SHAP waterfall chart (why this score?)
  3. Model performance tab (ROC curve, feature importance, confusion matrix)

Run: streamlit run dashboard/app.py
"""

import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit as st

# ── Load artifacts ──────────────────────────────────────────────────────────
BASE     = pathlib.Path(__file__).parent.parent
MODEL    = joblib.load(BASE / "model" / "model.joblib")
METRICS  = json.loads((BASE / "model" / "metrics.json").read_text())
FEATURES = json.loads((BASE / "model" / "feature_names.json").read_text())
EXPLAINER = shap.TreeExplainer(MODEL.named_steps["model"])

GRADE_COLORS = {"A": "#22c55e", "B": "#84cc16", "C": "#f59e0b", "D": "#f97316", "E": "#ef4444"}

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditGuard — AI Credit Risk Scorer",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ CreditGuard — AI Credit Risk Scoring")
st.caption("XGBoost + SHAP explainability · Goldman Sachs Risk Platform Demo")
st.divider()

tab_score, tab_model = st.tabs(["📋 Score a Borrower", "📊 Model Performance"])

# ─────────────────────────────────────────────────────────────────────────────
with tab_score:
    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.subheader("Borrower Profile")
        with st.form("borrower_form"):
            c1, c2 = st.columns(2)
            age               = c1.number_input("Age",               18, 80, 32)
            annual_income     = c2.number_input("Annual Income ($)", 10_000, 300_000, 65_000, step=1_000)
            credit_score      = c1.slider("Credit Score",            300, 850, 710)
            employment_years  = c2.number_input("Employment Years",  0.0, 40.0, 4.5, step=0.5)
            num_late_payments = c1.number_input("Late Payments",     0, 20, 1)
            utilization_rate  = c2.slider("Credit Utilization",      0.0, 1.0, 0.28, step=0.01,
                                          format="%.0f%%",
                                          help="Card balance / credit limit")
            num_credit_lines  = c1.number_input("# Credit Lines",    1, 30, 6)
            loan_amount       = c2.number_input("Loan Amount ($)",   500, 60_000, 12_000, step=500)
            loan_term_months  = c1.selectbox("Loan Term (months)", [12,24,36,48,60,72,84], index=2)
            home_ownership    = c2.selectbox("Home Ownership",       ["RENT","OWN","MORTGAGE"])
            loan_purpose      = c1.selectbox("Loan Purpose",
                ["DEBT_CONSOLIDATION","MEDICAL","HOME_IMPROVEMENT","EDUCATION","BUSINESS"])
            submitted = st.form_submit_button("🔍 Score Applicant", use_container_width=True)

    with col_result:
        if submitted:
            dti = (loan_amount / loan_term_months * 12) / annual_income if annual_income > 0 else 0
            row = pd.DataFrame([{
                "age": age, "annual_income": annual_income, "credit_score": credit_score,
                "employment_years": employment_years, "num_late_payments": num_late_payments,
                "utilization_rate": utilization_rate, "num_credit_lines": num_credit_lines,
                "loan_amount": loan_amount, "loan_term_months": loan_term_months,
                "debt_to_income": round(dti, 4), "home_ownership": home_ownership,
                "loan_purpose": loan_purpose,
            }])

            prob = float(MODEL.predict_proba(row)[0, 1])

            # Grade
            grade_map = {(0,.10):"A", (.10,.20):"B", (.20,.35):"C", (.35,.55):"D", (.55,2):"E"}
            grade = next(g for (lo,hi),g in grade_map.items() if lo <= prob < hi)
            color = GRADE_COLORS[grade]
            labels = {"A":"Very Low Risk","B":"Low Risk","C":"Moderate Risk",
                      "D":"High Risk","E":"Very High Risk"}

            st.subheader("Risk Assessment")
            st.markdown(
                f"<div style='background:{color}22;border:2px solid {color};"
                f"border-radius:12px;padding:20px;text-align:center'>"
                f"<div style='font-size:48px;font-weight:700;color:{color}'>{grade}</div>"
                f"<div style='font-size:18px;font-weight:500;color:{color}'>{labels[grade]}</div>"
                f"<div style='font-size:32px;margin-top:8px'>{prob:.1%}</div>"
                f"<div style='font-size:13px;color:#888'>Probability of Default</div></div>",
                unsafe_allow_html=True,
            )

            # SHAP waterfall
            st.subheader("Why this score?")
            transformed = MODEL.named_steps["prep"].transform(row)
            sv          = EXPLAINER.shap_values(transformed)[0]
            base_val    = float(EXPLAINER.expected_value)

            feat_names  = FEATURES["all"]
            top_n       = 8
            order       = np.argsort(np.abs(sv))[::-1][:top_n]
            names_top   = [feat_names[i] for i in order]
            vals_top    = [sv[i] for i in order]
            colors_top  = ["#ef4444" if v > 0 else "#22c55e" for v in vals_top]

            fig = go.Figure(go.Bar(
                x=vals_top[::-1], y=names_top[::-1],
                orientation="h",
                marker_color=colors_top[::-1],
            ))
            fig.update_layout(
                title="SHAP feature impact (red = increases default risk)",
                xaxis_title="SHAP value",
                height=340,
                margin=dict(l=0,r=0,t=40,b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fill in the borrower details and click **Score Applicant**.")

# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC",   f"{METRICS['roc_auc']:.4f}")
    m2.metric("Accuracy",  f"{METRICS['accuracy']:.4f}")
    m3.metric("Precision", f"{METRICS['precision']:.4f}")
    m4.metric("Recall",    f"{METRICS['recall']:.4f}")
    m5.metric("F1 Score",  f"{METRICS['f1']:.4f}")

    st.divider()
    st.subheader("Top Predictive Features (Mean |SHAP|)")

    fi = METRICS.get("feature_importance", {})
    fi_df = pd.DataFrame(list(fi.items())[:12], columns=["Feature", "Mean |SHAP|"])
    fig2 = px.bar(fi_df, x="Mean |SHAP|", y="Feature", orientation="h",
                  color="Mean |SHAP|", color_continuous_scale="Blues")
    fig2.update_layout(
        height=380, yaxis={"categoryorder":"total ascending"},
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Confusion Matrix")
    cm = METRICS["confusion_matrix"]
    cm_labels = ["Paid Back", "Defaulted"]
    fig3 = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=cm_labels, y=cm_labels, labels={"x":"Predicted","y":"Actual"},
    )
    fig3.update_layout(height=300, coloraxis_showscale=False,
                       paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)
