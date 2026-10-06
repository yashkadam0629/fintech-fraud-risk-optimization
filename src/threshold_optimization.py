"""
STAGE 3B - Financial threshold optimization.

The central idea of the project. A fraud model outputs a probability; the
business decision is where to intervene. 0.50 is a mathematical convention
with no commercial meaning: it implicitly assumes a missed fraud and an
annoyed customer cost exactly the same amount.

We instead define an explicit cost function

    TOTAL COST(t) = FN(t) x C_fn + FP(t) x C_fp

and sweep t from 0.01 to 0.99.

SIMULATED ASSUMPTIONS. C_fn and C_fp are stated assumptions, not measured
figures from any company. Nothing here is a real saving.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def build_threshold_grid(scores: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The fixed 0.01..0.99 grid, plus quantiles of the score distribution.

    A well-calibrated fraud model puts almost all of its probability mass below
    0.05, because fraud really is that rare. On a fixed 0.01 grid the entire
    interesting region collapses into four or five points, and the "optimal"
    threshold becomes an artifact of grid spacing. Adding score quantiles fixes
    that without abandoning the fixed grid, which stays in the output so the
    conventional 0.50 comparison remains available.
    """
    fixed = np.round(np.arange(config.THRESHOLD_GRID_START,
                               config.THRESHOLD_GRID_STOP + 1e-9,
                               config.THRESHOLD_GRID_STEP), 4)
    if scores is None or len(scores) == 0:
        return fixed, np.array(["fixed"] * len(fixed))

    quantiles = np.quantile(
        scores, np.linspace(config.QUANTILE_GRID_MIN, 0.99999,
                            config.QUANTILE_GRID_POINTS))
    quantiles = np.unique(np.clip(quantiles, 1e-6, 0.999999))
    quantiles = quantiles[~np.isin(np.round(quantiles, 4), fixed)]

    thresholds = np.concatenate([fixed, quantiles])
    source = np.array(["fixed"] * len(fixed) + ["quantile"] * len(quantiles))
    order = np.argsort(thresholds)
    return thresholds[order], source[order]


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray,
                    cost_fn: float = config.COST_FALSE_NEGATIVE,
                    cost_fp: float = config.COST_FALSE_POSITIVE,
                    use_quantile_grid: bool = True) -> pd.DataFrame:
    """Confusion matrix and cost at every threshold on the grid.

    Computed by sorting once and taking cumulative sums, rather than looping
    over the grid and re-scanning millions of rows each time.
    """
    thresholds, grid_source = build_threshold_grid(scores if use_quantile_grid else None)

    total_positive = int(y_true.sum())
    total_negative = int(len(y_true) - total_positive)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)

    # Number of rows scoring >= t, for every t on the grid.
    n_flagged = np.searchsorted(-sorted_scores, -thresholds, side="right")

    tp = np.where(n_flagged > 0, cumulative_tp[np.clip(n_flagged - 1, 0, None)], 0)
    fp = np.where(n_flagged > 0, cumulative_fp[np.clip(n_flagged - 1, 0, None)], 0)
    fn = total_positive - tp
    tn = total_negative - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(total_positive > 0, tp / total_positive, 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
        fpr = np.where(total_negative > 0, fp / total_negative, 0.0)

    return pd.DataFrame({
        "threshold": thresholds,
        "grid_source": grid_source,
        "TP": tp.astype(int), "TN": tn.astype(int),
        "FP": fp.astype(int), "FN": fn.astype(int),
        "precision": precision,
        "recall": recall,
        "fraud_capture_rate": recall,
        "false_positive_rate": fpr,
        "f1": f1,
        "legitimate_transactions_blocked": fp.astype(int),
        "fraud_missed": fn.astype(int),
        "alert_volume": (tp + fp).astype(int),
        "alert_rate": (tp + fp) / len(y_true),
        "cost_missed_fraud": fn * cost_fn,
        "cost_customer_friction": fp * cost_fp,
        "total_cost": fn * cost_fn + fp * cost_fp,
    })


def summarise_at(sweep: pd.DataFrame, threshold: float) -> dict:
    """Full reporting dictionary for one threshold on one sweep.

    Kept separate from `optimal_threshold` because the threshold is CHOSEN on
    the validation sweep and REPORTED on the test sweep. Re-optimising on the
    data you then report is a subtler form of the leakage this project is
    about, so the two operations must be separable.
    """
    best = sweep.iloc[(sweep["threshold"] - threshold).abs().argsort().iloc[0]]
    at_default = sweep.iloc[(sweep["threshold"] - 0.50).abs().argsort().iloc[0]]
    return _summary_from_rows(best, at_default)


def optimal_threshold(sweep: pd.DataFrame) -> dict:
    """The minimum-cost row, plus the comparisons that make it meaningful."""
    best = sweep.loc[sweep["total_cost"].idxmin()]
    at_default = sweep.iloc[(sweep["threshold"] - 0.50).abs().argsort().iloc[0]]
    return _summary_from_rows(best, at_default)


def _summary_from_rows(best: pd.Series, at_default: pd.Series) -> dict:

    total_fraud = int(best["TP"] + best["FN"])
    total_legit = int(best["TN"] + best["FP"])
    cost_fn = float(best["cost_missed_fraud"] / best["FN"]) if best["FN"] else np.nan
    cost_fp = float(best["cost_customer_friction"] / best["FP"]) if best["FP"] else np.nan

    # Reference points: intervene on nobody, intervene on everybody.
    do_nothing_cost = total_fraud * (cost_fn if np.isfinite(cost_fn) else config.COST_FALSE_NEGATIVE)
    block_all_cost = total_legit * (cost_fp if np.isfinite(cost_fp) else config.COST_FALSE_POSITIVE)

    return {
        "optimal_threshold": float(best["threshold"]),
        "total_cost_at_optimum": float(best["total_cost"]),
        "cost_missed_fraud": float(best["cost_missed_fraud"]),
        "cost_customer_friction": float(best["cost_customer_friction"]),
        "TP": int(best["TP"]), "FP": int(best["FP"]),
        "FN": int(best["FN"]), "TN": int(best["TN"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "fraud_capture_rate": float(best["fraud_capture_rate"]),
        "false_positive_rate": float(best["false_positive_rate"]),
        "legitimate_transactions_blocked": int(best["FP"]),
        "alert_volume": int(best["alert_volume"]),
        "alert_rate": float(best["alert_rate"]),
        "cost_at_threshold_0_50": float(at_default["total_cost"]),
        "cost_reduction_vs_0_50": float(at_default["total_cost"] - best["total_cost"]),
        "cost_reduction_vs_0_50_pct": float(
            100 * (at_default["total_cost"] - best["total_cost"]) / at_default["total_cost"]
        ) if at_default["total_cost"] else 0.0,
        "cost_if_no_intervention": float(do_nothing_cost),
        "cost_if_block_everything": float(block_all_cost),
        "disclaimer": (
            "SIMULATED / ESTIMATED UNDER ASSUMPTIONS. This threshold minimises "
            "simulated total cost for the stated cost assumptions on this "
            "evaluation sample only. It is not a universally optimal threshold "
            "and it is not a measured company saving."
        ),
    }


def derive_risk_bands(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Derive LOW / MEDIUM / HIGH boundaries from the cost model, not by hand.

    Two interventions with different costs imply two thresholds:

      BLOCK  - the expensive intervention (C_fp = blocking cost). Its
               minimum-cost threshold is the upper boundary.
      VERIFY - a cheaper step-up check (C_fp = verification cost). Because it
               annoys the customer less, its minimum-cost threshold sits
               lower, and that value becomes the lower boundary.

    So the MEDIUM band is exactly the region where intervening is worth it if
    the intervention is cheap, but not if it means a hard decline. That is a
    derived operating policy rather than three round numbers picked by eye.
    """
    block_sweep = threshold_sweep(y_true, scores,
                                  config.COST_FALSE_NEGATIVE, config.COST_FALSE_POSITIVE)
    verify_sweep = threshold_sweep(y_true, scores,
                                   config.COST_FALSE_NEGATIVE, config.COST_STEP_UP_VERIFICATION)

    block_threshold = float(block_sweep.loc[block_sweep["total_cost"].idxmin(), "threshold"])
    verify_threshold = float(verify_sweep.loc[verify_sweep["total_cost"].idxmin(), "threshold"])
    verify_threshold = min(verify_threshold, block_threshold)

    return {
        "high_risk_threshold_block": block_threshold,
        "medium_risk_threshold_verify": verify_threshold,
        "bands": [
            {"band": "LOW", "range": f"score < {verify_threshold:.2f}",
             "action": "allow without friction",
             "rationale": ("Below this score, the expected fraud loss prevented "
                           "is smaller than even the cheap verification cost.")},
            {"band": "MEDIUM", "range": f"{verify_threshold:.2f} <= score < {block_threshold:.2f}",
             "action": "step-up verification / manual review",
             "rationale": (f"Intervening pays off at a friction cost of "
                           f"${config.COST_STEP_UP_VERIFICATION:.0f} but not at "
                           f"${config.COST_FALSE_POSITIVE:.0f}.")},
            {"band": "HIGH", "range": f"score >= {block_threshold:.2f}",
             "action": "block / hard decline",
             "rationale": ("Expected fraud loss exceeds the full blocking cost "
                           "under the stated assumptions.")},
        ],
        "assumptions": {
            "cost_false_negative": config.COST_FALSE_NEGATIVE,
            "cost_false_positive_block": config.COST_FALSE_POSITIVE,
            "cost_false_positive_verify": config.COST_STEP_UP_VERIFICATION,
            "note": "SIMULATED assumptions, not measured company figures.",
        },
    }


def assign_risk_band(scores: np.ndarray, bands: dict) -> np.ndarray:
    low = bands["medium_risk_threshold_verify"]
    high = bands["high_risk_threshold_block"]
    return np.where(scores >= high, "HIGH", np.where(scores >= low, "MEDIUM", "LOW"))
