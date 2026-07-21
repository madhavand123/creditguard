"""
risk_grading.py — Single source of truth for converting a probability of
default into a letter grade (A-E).

Why this file exists:
  Previously api/main.py and dashboard/app.py each had their OWN copy of
  this grade-boundary logic. Two copies of the same business rule is a
  classic bug magnet — someone tweaks a threshold in one place during a
  demo and the API and dashboard silently disagree on a borrower's grade.
  Both now import from here instead.
"""

# (lower_bound_inclusive, upper_bound_exclusive) -> (grade, label, hex color)
RISK_GRADES = {
    (0.00, 0.10): ("A", "Very Low Risk",  "#22c55e"),
    (0.10, 0.20): ("B", "Low Risk",       "#84cc16"),
    (0.20, 0.35): ("C", "Moderate Risk",  "#f59e0b"),
    (0.35, 0.55): ("D", "High Risk",      "#f97316"),
    (0.55, 1.01): ("E", "Very High Risk", "#ef4444"),
}


def assign_grade(prob: float) -> dict:
    """Map a probability of default (0-1) to a risk grade dict."""
    for (lo, hi), (grade, label, color) in RISK_GRADES.items():
        if lo <= prob < hi:
            return {"grade": grade, "label": label, "color": color}
    # Fallback for prob == 1.0 or any float edge case
    return {"grade": "E", "label": "Very High Risk", "color": "#ef4444"}
