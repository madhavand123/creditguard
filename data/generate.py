"""
generate.py — Creates a realistic synthetic credit dataset.

We don't download anything. We model real-world credit default behaviour
using logistic probability driven by the features that actually matter to
lenders: credit score, debt-to-income ratio, late payment history, etc.
"""

import numpy as np
import pandas as pd

SEED = 42


def generate_credit_dataset(n_samples: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    age               = rng.integers(22, 65, n_samples)
    annual_income     = rng.lognormal(mean=11.0, sigma=0.5, size=n_samples).clip(20_000, 300_000)
    credit_score      = rng.normal(680, 80, n_samples).clip(300, 850).astype(int)
    employment_years  = rng.exponential(5, n_samples).clip(0, 30)
    num_late_payments = rng.poisson(1.2, n_samples).clip(0, 15)
    utilization_rate  = rng.beta(2, 5, n_samples)           # skewed low (most people < 0.5)
    num_credit_lines  = rng.integers(1, 18, n_samples)
    loan_amount       = rng.lognormal(9.5, 0.7, n_samples).clip(1_000, 60_000)
    loan_term_months  = rng.choice([12, 24, 36, 48, 60, 72, 84], n_samples)

    dti = (loan_amount / loan_term_months * 12) / annual_income
    dti = dti.clip(0, 1)

    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], n_samples, p=[0.40, 0.20, 0.40])
    loan_purpose   = rng.choice(
        ["DEBT_CONSOLIDATION", "MEDICAL", "HOME_IMPROVEMENT", "EDUCATION", "BUSINESS"],
        n_samples, p=[0.40, 0.15, 0.20, 0.15, 0.10]
    )

    # --- Default probability model (logistic) ---
    # Normalise credit_score to [0,1] so the coefficient is stable
    cs_norm = (credit_score - 300) / (850 - 300)   # 0 = worst, 1 = best

    log_odds = (
        0.0
        + (-4.0) * cs_norm                            # credit score is the biggest driver
        + 0.50   * num_late_payments
        + 3.50   * dti
        + 2.00   * utilization_rate
        + (-0.06)* employment_years
        + 0.50   * (home_ownership == "RENT").astype(float)
        + (-0.5) * (loan_purpose == "HOME_IMPROVEMENT").astype(float)
        + 0.80   * (loan_purpose == "BUSINESS").astype(float)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (rng.random(n_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        "age":               age,
        "annual_income":     annual_income.round(2),
        "credit_score":      credit_score,
        "employment_years":  employment_years.round(1),
        "num_late_payments": num_late_payments,
        "utilization_rate":  utilization_rate.round(4),
        "num_credit_lines":  num_credit_lines,
        "loan_amount":       loan_amount.round(2),
        "loan_term_months":  loan_term_months,
        "debt_to_income":    dti.round(4),
        "home_ownership":    home_ownership,
        "loan_purpose":      loan_purpose,
        "default":           default,
    })

    return df


if __name__ == "__main__":
    df = generate_credit_dataset()
    df.to_csv("data/credit_data.csv", index=False)
    print(f"Generated {len(df):,} rows  |  Default rate: {df['default'].mean():.1%}")
    print(df.head())
