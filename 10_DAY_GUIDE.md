# CreditGuard — 10-Day Learning Guide

**Goal:** understand every single line of this project from scratch, well enough to explain it, defend it, and modify it live in a Goldman Sachs interview.

**How to use this guide:** each day has Read → Understand → Do → Say. Don't skip "Do" — actually run the commands. Reading code you haven't run is how people get caught out when an interviewer asks "what happens if you change X?"

---

## Day 1 — What problem are we solving, and what does the data look like?

**Read:** `README.md` top to bottom, then `data/generate.py` in full.

**Understand:**
- A bank gives out loans. Some borrowers default (don't pay back). The bank wants to estimate that risk *before* approving the loan, not after.
- The model's job: given 12 facts about a borrower, output `P(default)` — a single number from 0 to 1.
- We don't have access to a real bank's loan book, so `generate.py` creates a synthetic dataset where the *relationships* between features and default are realistic even though no individual borrower is real. It does this with a hand-written formula (`log_odds`) that says: lower credit score → higher risk, more late payments → higher risk, and so on, then adds randomness on top so it isn't perfectly predictable (real defaults aren't either).
- `debt_to_income` isn't a raw input — it's *engineered*: `(loan_amount / loan_term_months * 12) / annual_income`. That's monthly payment burden as a fraction of annual income. Recognizing derived features vs raw inputs is a basic feature-engineering skill worth being able to point to.

**Do:**
```bash
python data/generate.py
```
Open the resulting `data/credit_data.csv` and look at the first 20 rows. Find one row with a high credit score and a low `default`, and one with a low credit score — confirm the pattern holds directionally (it's noisy, not deterministic).

**Key question to answer out loud:** Why does `num_late_payments` get a *positive* coefficient in `log_odds` (0.50) while `employment_years` gets a *negative* one (-0.06)? (Answer: more late payments increases default risk; more years employed is a stability signal that decreases it.)

---

## Day 2 — Machine learning basics (assumes zero prior ML)

**Understand:**
- **Supervised learning**: we have labeled examples — a borrower's features *and* whether they actually defaulted. The model learns the pattern from examples, then predicts on borrowers it's never seen.
- **Train/test split**: you never evaluate a model on data it trained on — that's like grading a student on the exact questions they saw the answer key for. We hold out 20% (4,000 of 20,000 rows) purely for evaluation.
- **XGBoost** = an ensemble of decision trees, built one at a time, where each new tree is trained specifically to correct the mistakes of the trees before it ("gradient boosting"). A single decision tree is easy to understand but weak; hundreds of trees correcting each other in sequence is both powerful and (with SHAP) still explainable.

**Read:** `model/train.py` lines 1–48 (imports, dataset generation, train/test split).

**Do:**
- Sketch a single decision tree on paper for credit risk, e.g.:
  ```
  if credit_score < 580: high risk
  elif num_late_payments > 3: medium risk
  else: low risk
  ```
- Now imagine 400 of these trees (see `n_estimators=400` in `train.py`), each one nudging the prediction based on what the previous trees got wrong. That's XGBoost.

**Key question:** why do we `stratify=y` in the train/test split? (Answer: so the ~25% default rate is preserved in both the train and test sets — otherwise a random split could accidentally put almost all the defaulters in one side.)

---

## Day 3 — The preprocessing pipeline

**Read:** `model/train.py` lines 50–69 (`ColumnTransformer`, `Pipeline`, `XGBClassifier` setup).

**Understand three things:**

1. **StandardScaler** (numeric features only): raw `annual_income` (~65,000) and `credit_score` (~710) live on wildly different scales. Scaling transforms each to roughly the same range (mean 0, std 1) using `(x - mean) / std`, computed from *training data only*. This helps many models train faster and more stably.

2. **OneHotEncoder** (categorical features): `home_ownership = "RENT"` becomes a 0/1 column per category (`home_ownership_RENT`, `home_ownership_OWN`, `home_ownership_MORTGAGE`) because models operate on numbers, not text. `handle_unknown="ignore"` means if a brand-new category shows up at inference time that wasn't in training, it doesn't crash — it just gets all zeros.

3. **`Pipeline`** chains preprocessing and the model into a single object. Calling `pipeline.fit(X_train, y_train)` fits the scaler's mean/std *and* trains the model, in one call, using only training data. Calling `pipeline.predict(row)` on new data automatically applies the *already-fitted* scaler before predicting — this is what prevents data leakage (the test set never influences the scaler's mean/std).

**Do:** Add `print(X_train.head())` and `print(X_train.dtypes)` near the top of `model/train.py` and rerun it — see the raw, unscaled values that go *into* the pipeline.

**Key question:** what would go wrong if you fit the `StandardScaler` on the *full* dataset (train + test) instead of just training data? (Answer: information from the test set — its mean and spread — leaks into preprocessing, giving an overly optimistic evaluation of how well the model generalizes.)

---

## Day 4 — Model evaluation metrics

**Read:** `model/train.py` lines 74–91.

**Understand each metric, and what it actually answers:**

- **Accuracy (0.718)**: "what fraction of predictions were correct overall?" Misleading on imbalanced data — with a 75% "no default" base rate, a model that *always* predicts "no default" gets 75% accuracy while being useless.
- **ROC-AUC (0.745)**: "if I pick a random defaulter and a random non-defaulter, how often does the model correctly rank the defaulter as riskier?" 0.5 = random guessing, 1.0 = perfect ranking. Threshold-independent — doesn't depend on where you draw the "risky" cutoff.
- **Precision (0.453)**: "of everyone flagged as high-risk, how many actually defaulted?" Low precision = rejecting good borrowers unnecessarily (false positives).
- **Recall (0.593)**: "of everyone who actually defaulted, how many did we catch?" Low recall = approving bad borrowers we should have caught (false negatives).
- **F1**: harmonic mean of precision and recall — one number when you care about both roughly equally.

**Interview-ready answer:** "I'd emphasize ROC-AUC here because it's threshold-independent and measures ranking quality, which is what a bank actually needs — order every applicant from least to most risky, then decide separately where to draw the approval line based on business appetite for risk."

**Do:** Open `model/metrics.json` and manually verify: `confusion_matrix` should be a 2x2 array `[[true_negatives, false_positives], [false_negatives, true_positives]]`. Compute precision and recall from those four numbers by hand and confirm they match the reported values.

---

## Day 5 — SHAP: why the model decided what it did

**Read:** `model/train.py` lines 93–116, and `api/main.py`'s `_predict_single` function.

**Understand:**
- SHAP = SHapley Additive exPlanations, based on a concept from cooperative game theory (Shapley values): fairly splitting "credit" for an outcome among contributors.
- For a single prediction, SHAP answers: "`credit_score` pushed this specific borrower's risk *down*, `debt_to_income` pushed it *up*, and here's by how much, relative to an average borrower." This is calculated per-prediction, not just once globally.
- Global feature importance (in `metrics.json`'s `feature_importance`) is the *average* of the absolute SHAP value across all test borrowers — "on average, how much does this feature move predictions."
- **Important nuance for this specific project:** these SHAP values are in log-odds space (the model's internal scoring space), not literal probability points. They're correct for *ranking* which features mattered most for a prediction, but don't expect them to arithmetically sum to the displayed probability — that's a common point of confusion worth pre-empting if asked.

**Analogy:** a sports team wins a game. How much did each player individually contribute to the win? SHAP distributes credit/blame across features the same way, in a mathematically fair way (each feature's contribution is its average marginal effect across all possible orderings of features).

**Do:**
- Run the dashboard (see Day 7) and score a borrower with `credit_score=400` vs the same borrower with `credit_score=800`. Watch how the SHAP bar chart changes — `credit_score` should flip from a large positive (risk-increasing) contribution to a large negative (risk-decreasing) one.

---

## Day 6 — FastAPI: the REST API

**Read:** `api/main.py` in full.

**Understand REST basics:**
- A REST API listens for HTTP requests and returns JSON. `POST /predict` sends borrower data in the request body and gets a risk score back. `GET /health` just confirms the server is alive — no request body needed.
- **Pydantic validation** happens automatically, before your code even runs:
  ```python
  class BorrowerInput(BaseModel):
      credit_score: int = Field(..., ge=300, le=850)
  ```
  Send `credit_score: 999` and FastAPI rejects it with an HTTP 422 error *before* it ever reaches the model — you never write `if credit_score > 850: raise error` by hand.
- The `@field_validator` methods do extra validation Pydantic's built-in `Field` can't express alone — e.g. checking `home_ownership` is one of exactly three allowed strings.

**Do:**
```bash
uvicorn api.main:app --reload --port 8000
```
- Open http://localhost:8000/docs and try `/predict` with the example payload from the README.
- Then send `credit_score: 999` and read the exact 422 error message you get back.
- Then hit `/model/info` and confirm `training_rows` shows `16000` — **this used to be a hardcoded, wrong value (`8000`) in this project**; it's now read live from `metrics.json`. If you're asked "walk me through a bug you found and fixed," this is a genuine, specific one to use.

---

## Day 7 — Streamlit dashboard

**Read:** `dashboard/app.py` in full.

**Understand:**
- Streamlit turns a plain Python script into an interactive web app — no HTML/CSS/JS required.
- `st.form(...)` groups all the input widgets together so the app only reruns and recomputes when you click the submit button, not on every keystroke.
- `st.plotly_chart(...)` renders an interactive Plotly figure inline.
- Notice the dashboard imports `assign_grade` from `risk_grading.py` — the exact same function `api/main.py` uses. This isn't an accident: it used to be a second, separately hand-maintained copy of the grading logic, which is a real risk (change a threshold in one file, forget the other, now the API and dashboard silently disagree on a borrower's grade). One shared function, imported in two places, removes that risk entirely.

**Do:**
```bash
streamlit run dashboard/app.py
```
- Score a few different borrower profiles.
- Change one borrower from `credit_score=800` to `credit_score=500` and watch the grade and SHAP chart both update.
- Open the "Model Performance" tab and check the feature importance chart against `model/metrics.json`'s `feature_importance` — they should match exactly, since they're reading the same file.

---

## Day 8 — Docker: making it deployable anywhere

**Read:** `Dockerfile`.

**Understand each line:**
```dockerfile
FROM python:3.11-slim              # Start from a minimal, clean Python image
WORKDIR /app                       # All subsequent commands run from /app
COPY requirements.txt .
RUN pip install -r requirements.txt   # Install dependencies first (better layer caching)
COPY . .                           # Copy the rest of the project files in
RUN python model/train.py          # Train the model during the image build itself
EXPOSE 8000                        # Documents which port the container listens on
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why `RUN python model/train.py` happens during the *build*, not at container start:** it bakes a freshly-trained model into the image itself, so every container started from this image is guaranteed to have a matching model + metrics, with no "someone forgot to train the model before deploying" failure mode.

**Why this matters to GS:** "works on my machine" isn't acceptable in production. A Docker container runs identically on a laptop, a GS build server, or a cloud instance — same OS, same dependency versions, same startup sequence, every time.

**Do:**
```bash
docker build -t creditguard .
docker run -p 8000:8000 creditguard
curl http://localhost:8000/health
```
Confirm it behaves exactly like the non-Docker version.

---

## Day 9 — System design view: tie it all together

**Draw this from memory** (don't copy it — actually redraw it on paper or a whiteboard until you can do it without looking):

```
Client (curl / Streamlit / browser)
          ↓  HTTP POST /predict
    FastAPI (api/main.py)
          ↓
    Pydantic validates input (rejects bad data with a 422, before the model ever sees it)
          ↓
    sklearn Pipeline:
      ├── StandardScaler (numeric)
      └── OneHotEncoder (categorical)
          ↓
    XGBoost model → P(default)
          ↓
    SHAP TreeExplainer → per-feature contribution for this specific prediction
          ↓
    risk_grading.assign_grade() → A/B/C/D/E
          ↓
    JSON response → client
```

**Talking points to have ready, verbatim in your own words:**
1. "The sklearn Pipeline prevents data leakage — the scaler's mean and std are fit only on training data."
2. "SHAP lets us explain *every* individual decision, not just the model in aggregate — critical for regulatory adverse-action requirements."
3. "The batch endpoint scores up to 100 borrowers per network call, which matters for latency when processing a loan book in bulk rather than one request per borrower."
4. "Docker eliminates environment drift between my laptop and wherever this actually runs."
5. "`scale_pos_weight` handles the class imbalance directly — without it, a model could hit 75% accuracy by just always predicting 'no default,' since that's roughly the base rate, and learn nothing useful."
6. "I found and fixed two real issues after the initial build: a hardcoded, stale `training_rows` value in the API, and duplicated risk-grading logic between the API and dashboard that I consolidated into one shared module."

---

## Day 10 — Polish, present, and prep the AI-usage question

**Polish checklist:**
- [ ] Push to GitHub with an incremental commit history (not one giant commit) — e.g. one commit for data gen, one for training, one for the API, one for the dashboard, one for fixes.
- [ ] Confirm the README's example `/predict` response actually matches what you get running it locally (it should — the dataset and model are both seeded).
- [ ] Add a screenshot or short screen-recording of the dashboard to the repo.
- [ ] Add your name and a one-line description to the GitHub repo's "About" section.
- [ ] Do one full clean-slate run: delete `model/model.joblib`, `model/metrics.json`, `model/feature_names.json`, and `data/credit_data.csv`, then run `python model/train.py` and confirm everything regenerates and the API/dashboard both still work.

**How to describe it on your resume:**
> "Built a credit default prediction system (XGBoost + SHAP + FastAPI + Streamlit) that scores loan applicants in ~12ms with explainable, per-feature risk factors and A–E risk grading, containerized with Docker for reproducible deployment."

**How to open the conversation about it in an interview:**
1. State the problem: "Banks need to estimate default risk before approving a loan, and they need to be able to explain that decision, not just produce a number."
2. Explain the system, briefly, top to bottom: data generation → training → API → dashboard.
3. Highlight the engineering choices, not just the ML: no data leakage (Pipeline), explainability (SHAP), reproducible deployment (Docker), and the fact that you found and fixed real bugs after the initial build rather than shipping the first working version.
4. Be ready for the AI-usage question (see below) and actually rehearse saying it out loud once, don't just read it silently. The goal is to sound like someone describing their own process, not reciting a script.
5. Connect it back to GS: "This mirrors the fundamentals behind real credit risk engines — feature-driven scoring, explainability for compliance, and reproducible deployment — just at portfolio-project scale instead of production scale."

---

### 🤖 How to answer "did you use AI to build this?"

If you're asked in an interview whether you used AI tools (Claude, ChatGPT, Copilot, etc.) to build this — **say yes, plainly, and then explain what you actually did with it.** Every strong engineer uses these tools now; what interviewers are actually evaluating is whether *you* understand what the AI-assisted code does, not whether you typed every character yourself. Dodging or downplaying it reads worse than being direct.

A genuine, defensible answer looks like this:

> "I used an AI coding assistant as an accelerator, the way I'd use a senior engineer's code review or a well-documented framework's boilerplate — not as a replacement for understanding the system. Concretely: I used it to scaffold the FastAPI routes and Pydantic schemas faster than writing them by hand, to get a working SHAP integration on the first try instead of debugging the API myself from the docs, and to help structure the Docker setup. But I made the actual modeling decisions myself — which features to include, how to handle class imbalance with `scale_pos_weight`, what the risk grade boundaries should be, and what the project's real limitations are. I also went back through the generated code afterward and fixed real bugs in it — for example, the API was reporting a hardcoded, stale training-set size instead of the real one, and the dashboard had its own separate copy of the risk-grading logic that could have silently drifted out of sync with the API's copy. I pulled that into one shared module."

**Why this answer works:**
- It's honest — no interviewer believes a from-scratch, fully-documented, containerized ML project with a 10-day study guide was written with zero AI assistance, and claiming otherwise is a credibility risk if pressed on any specific line.
- It's specific — naming exactly what you delegated (boilerplate, scaffolding) versus what you owned (modeling choices, debugging, architecture) shows you know the difference.
- It demonstrates the actual skill GS is hiring for: reviewing and correcting AI-generated code, not just accepting it. Be ready to explain the `training_rows` bug and the duplicated `risk_grading` logic if asked — those are real bugs from this exact project you can speak to with specifics.
- **Preparation tip:** don't just memorize this answer. Re-read `model/train.py` and `api/main.py` end to end until you could rewrite either from memory. If an interviewer asks you to explain any single line, you should be able to — regardless of who typed it first.
