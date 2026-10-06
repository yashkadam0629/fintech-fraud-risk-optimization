"""
STAGE 3C - Cost sensitivity analysis.

One threshold derived from one pair of cost assumptions is a fragile finding.
The defensible version is the map: how does the recommended operating point
move as the assumptions move?

That map is also the useful artifact commercially. A fraud lead rarely knows
the true cost of a blocked customer, but they can usually bound it, and this
turns "what is the threshold" into "here is the threshold for your range".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.threshold_optimization import threshold_sweep

logger = logging.getLogger(__name__)


def sensitivity_grid(y_true: np.ndarray, scores: np.ndarray,
                     fn_costs: list[float] | None = None,
                     fp_costs: list[float] | None = None) -> pd.DataFrame:
    """Minimum-cost threshold for every (FN cost, FP cost) combination.

    The confusion matrix at each threshold does not depend on the costs, so we
    compute the sweep once with unit costs and rescale. 25 scenarios cost
    almost nothing extra this way.
    """
    fn_costs = fn_costs or config.SENSITIVITY_FN_COSTS
    fp_costs = fp_costs or config.SENSITIVITY_FP_COSTS

    base = threshold_sweep(y_true, scores, cost_fn=1.0, cost_fp=1.0)
    rows = []
    for cost_fn in fn_costs:
        for cost_fp in fp_costs:
            total = base["FN"] * cost_fn + base["FP"] * cost_fp
            best_index = int(total.idxmin())
            best = base.loc[best_index]
            rows.append({
                "cost_false_negative": cost_fn,
                "cost_false_positive": cost_fp,
                "cost_ratio_fn_over_fp": cost_fn / cost_fp,
                "optimal_threshold": float(best["threshold"]),
                "total_cost": float(total.loc[best_index]),
                "TP": int(best["TP"]), "FP": int(best["FP"]),
                "FN": int(best["FN"]), "TN": int(best["TN"]),
                "fraud_capture_rate": float(best["fraud_capture_rate"]),
                "precision": float(best["precision"]),
                "false_positive_rate": float(best["false_positive_rate"]),
                "legitimate_transactions_blocked": int(best["FP"]),
                "fraud_missed": int(best["FN"]),
                "alert_rate": float(best["alert_rate"]),
            })
    return pd.DataFrame(rows)


def threshold_matrix(grid: pd.DataFrame, value: str = "optimal_threshold") -> pd.DataFrame:
    """Pivot the grid into the matrix that becomes the heatmap."""
    return grid.pivot(index="cost_false_negative",
                      columns="cost_false_positive",
                      values=value).sort_index(ascending=False)


def summarise_sensitivity(grid: pd.DataFrame) -> dict:
    """Plain-language read of what the grid shows, derived from the numbers."""
    spread = float(grid["optimal_threshold"].max() - grid["optimal_threshold"].min())
    by_ratio = (grid.groupby("cost_ratio_fn_over_fp")["optimal_threshold"]
                .mean().sort_index())
    direction = "falls" if by_ratio.iloc[-1] < by_ratio.iloc[0] else "rises"

    capture_range = (float(grid["fraud_capture_rate"].min()),
                     float(grid["fraud_capture_rate"].max()))
    blocked_range = (int(grid["legitimate_transactions_blocked"].min()),
                     int(grid["legitimate_transactions_blocked"].max()))

    return {
        "scenarios_tested": int(len(grid)),
        "threshold_min": float(grid["optimal_threshold"].min()),
        "threshold_max": float(grid["optimal_threshold"].max()),
        "threshold_spread": spread,
        "threshold_median": float(grid["optimal_threshold"].median()),
        "fraud_capture_rate_range": capture_range,
        "legitimate_blocked_range": blocked_range,
        "direction_with_rising_fn_fp_ratio": direction,
        "interpretation": (
            f"Across {len(grid)} cost scenarios the minimum-cost threshold ranges "
            f"from {grid['optimal_threshold'].min():.2f} to "
            f"{grid['optimal_threshold'].max():.2f}. As missed fraud becomes more "
            f"expensive relative to customer friction, the threshold {direction}, "
            f"because the model should intervene more readily when being wrong in "
            f"the other direction costs more. The operating point is therefore a "
            f"function of business economics, not a fixed property of the model."
        ),
        "disclaimer": "SIMULATED / ESTIMATED UNDER ASSUMPTIONS.",
    }
