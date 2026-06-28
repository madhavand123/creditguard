"""
train.py — Trains an XGBoost credit default classifier.

Pipeline:
  Raw CSV → preprocessing → XGBoost → evaluation → save artifacts

Outputs (saved to model/):
  model.joblib       — trained pipeline (preprocessor + XGB)
  metrics.json       — AUC, accuracy, precision, recall, F1
  feature_names.json — ordered feature list for the API
"""

import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import shap

from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.preprocessing    import StandardScaler, OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBClassifier

from data.generate import generate_credit_dataset


# ── 1. Load / generate data ────────────────────────────────────────────────
print("Generating dataset …")
df = generate_credit_dataset(n_samples=20_000)
df.to_csv("data/credit_data.csv", index=False)

TARGET   = "default"
CAT_COLS = ["home_ownership", "loan_purpose"]
NUM_COLS = [c for c in df.columns if c not in CAT_COLS + [TARGET]]

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Preprocessing ────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])

# ── 3. XGBoost model ────────────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators      = 400,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = (y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder = False,
    eval_metric       = "auc",
    random_state      = 42,
)

pipeline = Pipeline([("prep", preprocessor), ("model", xgb)])

print("Training XGBoost …")
pipeline.fit(X_train, y_train)

# ── 4. Evaluation ───────────────────────────────────────────────────────────
y_pred      = pipeline.predict(X_test)
y_prob      = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall":    round(recall_score(y_test, y_pred), 4),
    "f1":        round(f1_score(y_test, y_pred), 4),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "default_rate": round(float(y.mean()), 4),
    "test_size": len(y_test),
}
print("\n── Model Performance ─────────────────────")
for k, v in metrics.items():
    if k != "confusion_matrix":
        print(f"  {k:15s}: {v}")

# ── 5. SHAP feature importance ──────────────────────────────────────────────
print("\nComputing SHAP values …")
X_test_transformed = pipeline.named_steps["prep"].transform(X_test)

# Get feature names after one-hot encoding
ohe_features = list(
    pipeline.named_steps["prep"]
    .named_transformers_["cat"]
    .get_feature_names_out(CAT_COLS)
)
all_feature_names = NUM_COLS + ohe_features

explainer   = shap.TreeExplainer(pipeline.named_steps["model"])
shap_values = explainer.shap_values(X_test_transformed)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = dict(
    zip(all_feature_names, mean_abs_shap.round(5).tolist())
)
feature_importance = dict(
    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
)

metrics["feature_importance"] = feature_importance

# ── 6. Save artifacts ───────────────────────────────────────────────────────
pathlib.Path("model").mkdir(exist_ok=True)
joblib.dump(pipeline, "model/model.joblib")

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("model/feature_names.json", "w") as f:
    json.dump({"numeric": NUM_COLS, "categorical": CAT_COLS,
               "all": all_feature_names}, f, indent=2)

print("\n✓ Artifacts saved to model/")
print(f"  ROC-AUC : {metrics['roc_auc']}")
print(f"  Accuracy: {metrics['accuracy']}")
