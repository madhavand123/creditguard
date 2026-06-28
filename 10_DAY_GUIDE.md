# CreditGuard — 10-Day Learning Guide

Goal: understand every single line of this project well enough to explain it in a GS interview.

---

## Day 1 — What problem are we solving?

**Read:** README.md top to bottom.

**Understand:**
- A bank gives out loans. Some people default (don't pay back). The bank wants to know *before* giving the loan.
- We build a model: given 12 features about a borrower, output P(default) — a number from 0 to 1.
- If P > 0.35 → risky loan. Grade D or E. Maybe reject, or charge higher interest.

**Do:**
- Open `data/generate.py`. Read the `generate_credit_dataset()` function.
- Understand each feature: what is `debt_to_income`? What is `utilization_rate`?
- Run `python data/generate.py` and look at the CSV it produces.

**Key question to answer:** Why does a higher `num_late_payments` increase default risk?

---

## Day 2 — Machine Learning basics (no prior ML needed)

**Understand:**
- Supervised learning: we have labeled examples (borrower + whether they defaulted).
- The model learns the pattern from 16,000 examples, then predicts on new unseen borrowers.
- XGBoost = an ensemble of decision trees. Each tree corrects the errors of the previous one (gradient boosting).

**Read:**
- `model/train.py` lines 1–40 (imports and data split).
- Understand `train_test_split`: 80% of data to train on, 20% to evaluate on (never seen during training).

**Do:**
- Draw a simple decision tree on paper for credit risk:
  - If credit_score < 580 → high risk
  - Else if num_late_payments > 3 → medium risk
  - Else → low risk
- XGBoost makes hundreds of these trees and combines them.

---

## Day 3 — Preprocessing pipeline

**Read:** `model/train.py` lines 40–60 (ColumnTransformer, Pipeline).

**Understand two things:**

1. **StandardScaler** (for numeric features):
   - Raw: income = 65,000, credit_score = 710 → very different scales.
   - After scaling: both become roughly -3 to +3. Model trains faster and more stably.
   - Formula: `(x - mean) / std`

2. **OneHotEncoder** (for categorical features):
   - `home_ownership = "RENT"` → `[1, 0, 0]`
   - `home_ownership = "OWN"`  → `[0, 1, 0]`
   - `home_ownership = "MORTGAGE"` → `[0, 0, 1]`
   - Models can't handle text, so we convert to numbers.

3. **Pipeline** chains preprocessing + model into one object. When you call `pipeline.predict(row)`, it automatically preprocesses first, then predicts. Clean and no data leakage.

**Do:** Add a `print(X_train.head())` and `print(X_train.dtypes)` in train.py and run it.

---

## Day 4 — Model evaluation metrics

**Read:** `model/train.py` lines 60–80.

**Understand each metric:**

- **Accuracy** (0.72): 72% of predictions are correct. Sounds good, but misleading if classes are imbalanced.
- **ROC-AUC** (0.75): Better. At 0.5 = random, at 1.0 = perfect. Measures ranking quality — does the model rank defaulters higher than non-defaulters?
- **Precision** (0.46): Of everyone the model says "will default", 46% actually do. False positives = approving someone who defaults.
- **Recall** (0.59): Of all actual defaulters, we catch 59%. False negatives = we miss a defaulter.
- **F1**: Harmonic mean of precision + recall. Use when both matter.

**GS interview answer:** "I prioritised ROC-AUC because it's threshold-independent and tells us how well the model *ranks* borrowers — which is what a bank cares about: rank everyone from least to most risky."

---

## Day 5 — SHAP (why the model decided what it did)

**Read:** `model/train.py` lines 80–100 and `api/main.py` lines 60–80.

**Understand:**
- SHAP = SHapley Additive exPlanations (from game theory).
- For each prediction, SHAP tells you: "credit_score pushed the default probability DOWN by 0.31, but num_late_payments pushed it UP by 0.09."
- This is legally critical: regulators require banks to explain *why* a loan was denied.

**Analogy:**
- A football team wins. How much did each player contribute? SHAP fairly distributes the "credit" among features.

**Do:**
- Read `dashboard/app.py` lines 80–105 (SHAP bar chart section).
- Run the dashboard and score a borrower with credit_score=400 vs credit_score=800. See how the SHAP chart changes.

---

## Day 6 — FastAPI (building the REST API)

**Read:** `api/main.py` all of it.

**Understand REST API basics:**
- A REST API listens for HTTP requests and returns JSON responses.
- `POST /predict` → you send borrower data, you get back a risk score.
- `GET /health` → just confirms the server is running.

**Understand Pydantic validation:**
```python
class BorrowerInput(BaseModel):
    credit_score: int = Field(..., ge=300, le=850)
```
- If someone sends `credit_score: 999`, FastAPI automatically rejects it with a 422 error.
- You never need to write `if credit_score > 850: raise error` manually.

**Do:**
- Start the API: `uvicorn api.main:app --reload --port 8000`
- Open http://localhost:8000/docs
- Try the `/predict` endpoint with the example payload from the README.
- Try sending a bad value (credit_score: 999) and see what error you get.

---

## Day 7 — Streamlit dashboard

**Read:** `dashboard/app.py` all of it.

**Understand:**
- Streamlit turns Python scripts into interactive web apps. No HTML/JS needed.
- `st.slider()` / `st.number_input()` → interactive UI widgets.
- `st.plotly_chart()` → renders a Plotly chart inline.
- `st.form()` groups inputs and only runs the code when you click "Score Applicant".

**Do:**
- Run `streamlit run dashboard/app.py`
- Score several borrowers. Notice how the SHAP chart changes based on inputs.
- Change a borrower from credit_score=800 to credit_score=500. What happens to the grade?

---

## Day 8 — Docker (making it deployable anywhere)

**Read:** `Dockerfile`.

**Understand each line:**
```dockerfile
FROM python:3.11-slim          # Start from a clean Python image
WORKDIR /app                   # All commands run from /app
COPY requirements.txt .
RUN pip install -r requirements.txt   # Install dependencies
COPY . .                       # Copy all project files
RUN python model/train.py      # Train the model during build
EXPOSE 8000                    # Tell Docker this port is used
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why Docker matters to GS:**
- "Works on my machine" is not acceptable in production.
- Docker containers run identically on your laptop, on a GS server, or on AWS.
- Every GS microservice runs in a container.

**Do:**
- Install Docker Desktop.
- Run: `docker build -t creditguard .`
- Run: `docker run -p 8000:8000 creditguard`
- Call the API at http://localhost:8000/docs — it works exactly the same as without Docker.

---

## Day 9 — Tie everything together (system design view)

**Draw this architecture diagram:**

```
Client (curl / Streamlit / browser)
          ↓  HTTP POST /predict
    FastAPI server (api/main.py)
          ↓
    Pydantic validates input
          ↓
    sklearn Pipeline:
      ├── StandardScaler (numeric)
      └── OneHotEncoder (categorical)
          ↓
    XGBoost model → P(default)
          ↓
    SHAP explainer → feature impacts
          ↓
    Risk grade assignment (A/B/C/D/E)
          ↓
    JSON response → client
```

**GS interview talking points to memorise:**
1. "I used a sklearn Pipeline so there's no data leakage between train and test — the scaler's mean/std is fit only on training data."
2. "SHAP ensures we can explain every decision — critical for ECOA and GDPR compliance."
3. "The batch endpoint lets us score 100 borrowers in one network call — reduces latency for bulk loan processing."
4. "Docker means zero environment drift between development and production."
5. "scale_pos_weight handles class imbalance — default rate is ~25%, so without this, the model would just predict 'no default' 100% of the time and get 75% accuracy by doing nothing."

---

## Day 10 — Polish and present

**Polish checklist:**
- [ ] Push project to GitHub with a clean commit history (not just one big commit)
- [ ] Update README with your actual metrics and a screenshot of the dashboard
- [ ] Add a `demo.gif` or screenshot to the GitHub repo
- [ ] Add your name and a brief description to the repo About section
- [ ] Test the full pipeline once from scratch on a clean folder (no existing model.joblib)

**How to describe it on your resume:**
> "Built a credit default prediction API (XGBoost + SHAP + FastAPI) that scores loan applicants in <20ms with explainable risk grades (A–E), containerized via Docker and served through a Streamlit monitoring dashboard."

**How to talk about it in an interview:**
1. Start with the problem: "Banks need to know if someone will repay a loan before approving it."
2. Explain the system: "I built an end-to-end ML pipeline — data generation, XGBoost model, REST API, and a dashboard."
3. Highlight the engineering: "I used sklearn Pipelines to prevent data leakage, SHAP for regulatory explainability, and Docker for reproducible deployment."
4. Connect to GS: "This directly maps to Goldman's risk platform work — the same fundamentals behind credit risk engines in large financial institutions."
