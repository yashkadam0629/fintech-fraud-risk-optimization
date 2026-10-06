"""
STAGE 3 - Model training.

Two decisions here carry more weight than the algorithm choice:

1. TIME-AWARE SPLIT.
   A random split would put transaction #500 of a destination account in the
   training set and transaction #499 in the test set. Every behavioural
   feature is cumulative, so the model would be scored on a period it had
   already partly seen. Worse, it would be evaluated on a fraud population
   that did not exist yet in production terms. Splitting on `step` reproduces
   the only situation that matters: train on the past, predict the future.

2. LEAKAGE-SAFE FEATURE SET.
   Balance columns are excluded. See the Stage 1 leakage audit for the
   measured justification. A separate, clearly labelled leakage demonstration
   model can be trained on them to show the inflation they cause.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)

try:  # optional dependency
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False


def time_aware_split(df: pd.DataFrame,
                     train_fraction: float = config.TRAIN_FRACTION,
                     validation_fraction: float = config.VALIDATION_FRACTION) -> dict:
    """Split chronologically on `step`, never cutting a step across two sets.

    Returns boolean masks plus the step boundaries, so the split is auditable
    and reproducible rather than hidden inside a train_test_split call.
    """
    per_step = df.groupby("step", observed=True).size().sort_index()
    cumulative = per_step.cumsum() / per_step.sum()

    train_end = int(cumulative[cumulative <= train_fraction].index.max())
    val_cutoff = train_fraction + validation_fraction
    val_candidates = cumulative[cumulative <= val_cutoff]
    val_end = int(val_candidates.index.max()) if len(val_candidates) else train_end

    masks = {
        "train": df["step"] <= train_end,
        "validation": (df["step"] > train_end) & (df["step"] <= val_end),
        "test": df["step"] > val_end,
    }
    summary = {
        "train_step_range": [int(df["step"].min()), train_end],
        "validation_step_range": [train_end + 1, val_end],
        "test_step_range": [val_end + 1, int(df["step"].max())],
    }
    for name, mask in masks.items():
        subset = df.loc[mask]
        summary[name] = {
            "rows": int(mask.sum()),
            "row_pct": float(100 * mask.mean()),
            "fraud": int(subset[config.TARGET].sum()),
            "fraud_rate_pct": float(100 * subset[config.TARGET].mean()) if len(subset) else 0.0,
        }
    logger.info("Split -> train %s | val %s | test %s rows",
                f"{summary['train']['rows']:,}",
                f"{summary['validation']['rows']:,}",
                f"{summary['test']['rows']:,}")

    if summary["validation"]["fraud"] == 0 or summary["test"]["fraud"] == 0:
        logger.warning("A split contains zero fraud cases. Metrics on that split "
                       "will be undefined or meaningless.")
    return {"masks": masks, "summary": summary}


def build_models(fast: bool = False, class_weight=None) -> dict:
    """Two core models, plus XGBoost when the package is installed.

    Logistic regression is scaled because its coefficients are scale-sensitive;
    tree models are not, so they take the raw matrix.

    On class imbalance: by default nothing is rebalanced. Re-weighting or
    resampling the rare class inflates predicted probabilities, and this
    project spends Stage 3B deriving a threshold from an explicit cost model,
    which only means something if 0.04 really is a 4% estimate. Imbalance is
    handled at the decision layer instead. Pass class_weight="balanced" to see
    what the alternative does; expect every cost-optimal threshold to migrate
    toward 0.99.
    """
    logreg_params = {**config.LOGREG_PARAMS, "class_weight": class_weight}
    forest_params = {**(config.RANDOM_FOREST_PARAMS_FAST if fast
                        else config.RANDOM_FOREST_PARAMS),
                     "class_weight": class_weight}
    models: dict = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**logreg_params)),
        ]),
        "random_forest": RandomForestClassifier(**forest_params),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(**config.XGBOOST_PARAMS)
    else:
        logger.info("xgboost not installed; continuing with logistic regression "
                    "and random forest. This is not a problem for the project.")
    return models


def fit_models(models: dict, X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    """Fit each model, surviving individual failures."""
    fitted = {}
    for name, model in models.items():
        logger.info("Training %s on %s rows x %s features",
                    name, f"{len(X_train):,}", X_train.shape[1])
        try:
            if name == "xgboost":
                positive = max(int(y_train.sum()), 1)
                negative = int(len(y_train) - positive)
                model.set_params(scale_pos_weight=negative / positive)
            model.fit(X_train, y_train)
            fitted[name] = model
        except Exception:
            logger.exception("Training failed for %s; skipping it.", name)
    if not fitted:
        raise RuntimeError("No model trained successfully.")
    return fitted


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Fraud probability for the positive class."""
    return model.predict_proba(X)[:, 1]


def train_leakage_demo(X_train: pd.DataFrame, y_train: np.ndarray,
                       fast: bool = False) -> RandomForestClassifier:
    """LEAKAGE DEMONSTRATION MODEL - NOT VALID FOR DEPLOYMENT.

    Trained on the excluded balance columns to quantify how much apparent
    performance target leakage manufactures. Its scores must never be reported
    as legitimate model performance.
    """
    model = RandomForestClassifier(
        **(config.RANDOM_FOREST_PARAMS_FAST if fast else config.RANDOM_FOREST_PARAMS))
    model.fit(X_train, y_train)
    return model
