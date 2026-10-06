"""
STAGE 3D - Interpretability.

Language discipline throughout this module: feature importance describes what
the MODEL used, not what CAUSED the fraud. A feature can rank highly because
it is a genuine signal, because it correlates with one, or because of an
artifact of the data-generating process. We therefore say "contributed to the
model's score" and never "caused".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

try:  # optional dependency
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False


def model_feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Built-in importance: impurity decrease for trees, |coefficient| for LR."""
    estimator = model[-1] if hasattr(model, "steps") else model

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "impurity_decrease"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
        kind = "absolute_standardised_coefficient"
    else:
        return None

    return (pd.DataFrame({"feature": feature_names, "importance": values,
                          "importance_type": kind})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def permutation_feature_importance(model, X: pd.DataFrame, y: np.ndarray,
                                   sample_size: int = 50_000,
                                   n_repeats: int = 3,
                                   random_state: int = 42) -> pd.DataFrame | None:
    """Shuffle one feature, see how much PR-AUC drops.

    More trustworthy than impurity importance, which is biased toward
    high-cardinality features. Run on a sample because it refits nothing but
    re-scores the data once per feature per repeat.
    """
    try:
        rng = np.random.default_rng(random_state)
        if len(X) > sample_size:
            # Keep every fraud case; sample the legitimate majority. Otherwise
            # a 0.1% positive rate leaves too few positives to measure against.
            positive_idx = np.flatnonzero(y == 1)
            negative_idx = np.flatnonzero(y == 0)
            keep_negative = rng.choice(
                negative_idx, size=max(sample_size - len(positive_idx), 1), replace=False)
            idx = np.sort(np.concatenate([positive_idx, keep_negative]))
            X, y = X.iloc[idx], y[idx]

        if y.sum() == 0:
            return None

        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state,
            scoring=lambda est, X_, y_: average_precision_score(
                y_, est.predict_proba(X_)[:, 1]),
        )
        return (pd.DataFrame({"feature": X.columns,
                              "pr_auc_drop_when_shuffled": result.importances_mean,
                              "std": result.importances_std})
                .sort_values("pr_auc_drop_when_shuffled", ascending=False)
                .reset_index(drop=True))
    except Exception:
        logger.exception("Permutation importance failed; continuing without it.")
        return None


def shap_summary(model, X: pd.DataFrame, sample_size: int = 2_000) -> pd.DataFrame | None:
    """Mean |SHAP value| per feature, if the shap package is installed."""
    if not SHAP_AVAILABLE:
        logger.info("shap not installed; using permutation importance instead.")
        return None
    try:
        estimator = model[-1] if hasattr(model, "steps") else model
        sample = X.sample(min(sample_size, len(X)), random_state=42)
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            values = values[1]
        if values.ndim == 3:
            values = values[:, :, 1]
        return (pd.DataFrame({"feature": sample.columns,
                              "mean_abs_shap": np.abs(values).mean(axis=0)})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True))
    except Exception:
        logger.exception("SHAP failed; continuing without it.")
        return None


def explain_examples(scored: pd.DataFrame, feature_names: list[str],
                     importance: pd.DataFrame | None,
                     n_examples: int = 10, top_signals: int = 5) -> list[dict]:
    """Worked examples for the highest-scoring transactions.

    For each, report the contextual facts an analyst needs plus the features
    that the model weights most heavily and where this transaction sits
    relative to the population on those features.
    """
    ranked_features = (importance["feature"].tolist()[: top_signals * 3]
                       if importance is not None else feature_names[: top_signals * 3])
    ranked_features = [f for f in ranked_features if f in scored.columns][:top_signals]

    population_median = scored[ranked_features].median()
    examples = []
    top = scored.nlargest(n_examples, "fraud_probability")

    for _, row in top.iterrows():
        signals = []
        for feature in ranked_features:
            value = row[feature]
            median = population_median[feature]
            if pd.isna(value):
                continue
            comparison = "above" if value > median else ("below" if value < median else "at")
            signals.append({
                "feature": feature,
                "value": float(value),
                "population_median": float(median),
                "relative_position": comparison,
            })
        examples.append({
            "transaction_id": int(row.get("transaction_id", -1)),
            "step": int(row["step"]),
            "type": str(row["type"]),
            "amount": float(row["amount"]),
            "fraud_probability": float(row["fraud_probability"]),
            "risk_band": str(row.get("risk_band", "")),
            "actual_label": int(row.get("isFraud", -1)),
            "contributing_signals": signals,
            "wording_note": ("These features contributed to the model's score. "
                             "They are associated with the prediction and must not "
                             "be described as causes of fraud."),
        })
    return examples
