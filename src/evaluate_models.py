"""
STAGE 3 - Evaluation.

Two families of metric, kept apart on purpose:

THRESHOLD-FREE (ROC-AUC, PR-AUC) measure how well the model *ranks*
transactions. They answer "is the scoring any good at all".

THRESHOLD-BASED (precision, recall, F1, confusion matrix) measure a specific
operating decision. They answer "what happens if we intervene at 0.5", which
is a business choice, not a property of the model. Stage 3B replaces that
arbitrary 0.5 with a cost-derived value.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils import binary_metrics

logger = logging.getLogger(__name__)


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    prevalence = float(np.mean(y_true))
    if not (0 < y_true.sum() < len(y_true)):
        return {"roc_auc": None, "pr_auc": None, "pr_auc_no_skill_baseline": prevalence}
    pr_auc = float(average_precision_score(y_true, scores))
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": pr_auc,
        "pr_auc_no_skill_baseline": prevalence,
        "pr_auc_lift_over_no_skill": float(pr_auc / prevalence) if prevalence else None,
    }


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    result = ranking_metrics(y_true, scores)
    result["threshold"] = float(threshold)
    result.update(binary_metrics(y_true, (scores >= threshold).astype(int)))
    return result


def top_k_capture(y_true: np.ndarray, scores: np.ndarray,
                  fractions=(0.001, 0.005, 0.01, 0.02, 0.05)) -> list[dict]:
    """Fraud captured if the team can only review the top K% of scores.

    Real fraud teams have a fixed review headcount, so this is often the most
    actionable framing of model quality: "with capacity for 1% of traffic,
    what share of fraud lands in the queue".
    """
    order = np.argsort(-scores, kind="stable")
    ordered_labels = y_true[order]
    captured = np.cumsum(ordered_labels)
    total_fraud = max(int(y_true.sum()), 1)
    n = len(y_true)
    rows = []
    for fraction in fractions:
        k = max(int(fraction * n) - 1, 0)
        rows.append({
            "review_fraction": float(fraction),
            "transactions_reviewed": int(k + 1),
            "fraud_captured": int(captured[k]),
            "fraud_capture_rate": float(captured[k] / total_fraud),
            "precision_in_queue": float(captured[k] / (k + 1)),
        })
    return rows


def compare_models(y_true: np.ndarray, score_map: dict[str, np.ndarray]) -> pd.DataFrame:
    """Rank models by PR-AUC, which is the right primary metric here."""
    rows = []
    for name, scores in score_map.items():
        row = {"model": name}
        row.update(ranking_metrics(y_true, scores))
        row.update({f"at_0.5_{k}": v for k, v in
                    binary_metrics(y_true, (scores >= 0.5).astype(int)).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False, na_position="last")


def curve_points(y_true: np.ndarray, scores: np.ndarray, max_points: int = 500) -> dict:
    """Thinned ROC and PR curves, small enough to store and plot."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    def thin(array: np.ndarray) -> list:
        if len(array) <= max_points:
            return [float(v) for v in array]
        idx = np.linspace(0, len(array) - 1, max_points).astype(int)
        return [float(v) for v in array[idx]]

    return {"fpr": thin(fpr), "tpr": thin(tpr),
            "precision": thin(precision), "recall": thin(recall)}
