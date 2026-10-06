"""
STAGE 4 - Dashboard data preparation.

Power BI should not recompute analytics. It should present decisions that were
already made and validated in Python. This module writes one flat file per
dashboard element, each with a stable schema.

Files produced in outputs/dashboard_data/:
  scored_transactions.csv   row-level scored test set (Pages 1, 2, 3)
  kpi_summary.csv           single-row executive KPIs (Page 1)
  threshold_cost_curve.csv  threshold vs cost and confusion counts (Page 1)
  cost_sensitivity.csv      25-scenario grid (Page 1 detail)
  fraud_by_type.csv         fraud and false positives per transaction type
  fraud_over_time.csv       per-step trend of volume, fraud and alerts
  false_positive_profile.csv false positives by type, amount band, time
  risk_band_definition.csv  the derived bands and their rationale
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

AMOUNT_BANDS = [0, 1_000, 10_000, 50_000, 200_000, 1_000_000, np.inf]
AMOUNT_BAND_LABELS = ["0-1k", "1k-10k", "10k-50k", "50k-200k", "200k-1M", "1M+"]

CONTEXT_COLUMNS = [
    "transaction_id", "step", "sim_day", "hour_of_day", "type", "amount",
    "orig_id", "dest_id", "dest_is_merchant",
    "dest_prior_txn_count", "dest_prior_unique_senders",
    "dest_txn_prev_24h", "dest_txn_prev_168h", "dest_txn_prev_1h",
    "pair_prior_txn_count", "pair_is_new", "dest_is_new",
    "orig_prior_txn_count", "orig_amount_to_prior_avg",
    "amount_to_type_prior_avg",
    "fraud_probability", "risk_band", "predicted_flag", "isFraud",
    "outcome_class",
]


def build_scored_table(df_test: pd.DataFrame, scores: np.ndarray,
                       risk_bands: np.ndarray, threshold: float) -> pd.DataFrame:
    """Row-level output: one line per transaction with score, band and outcome."""
    scored = df_test.copy()
    scored["transaction_id"] = np.arange(1, len(scored) + 1)
    scored["fraud_probability"] = scores
    scored["risk_band"] = risk_bands
    scored["predicted_flag"] = (scores >= threshold).astype(int)

    actual = scored[config.TARGET].to_numpy()
    predicted = scored["predicted_flag"].to_numpy()
    scored["outcome_class"] = np.select(
        [(predicted == 1) & (actual == 1),
         (predicted == 1) & (actual == 0),
         (predicted == 0) & (actual == 1)],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )
    scored["amount_band"] = pd.cut(scored["amount"], bins=AMOUNT_BANDS,
                                   labels=AMOUNT_BAND_LABELS, right=False)
    return scored


def _write(frame: pd.DataFrame, name: str, directory: Path) -> Path:
    path = directory / name
    frame.to_csv(path, index=False)
    logger.info("  wrote %s (%s rows)", name, f"{len(frame):,}")
    return path


def export_all(scored: pd.DataFrame, sweep: pd.DataFrame, sensitivity: pd.DataFrame,
               optimum: dict, bands: dict, ranking: dict,
               directory: Path | None = None) -> dict[str, Path]:
    directory = Path(directory) if directory else config.DASHBOARD_DIR
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # ---- row level -------------------------------------------------------
    columns = [c for c in CONTEXT_COLUMNS if c in scored.columns] + ["amount_band"]
    queue = scored[columns].sort_values("fraud_probability", ascending=False)
    written["scored_transactions"] = _write(queue, "scored_transactions.csv", directory)

    # ---- executive KPIs --------------------------------------------------
    total = len(scored)
    fraud_rows = scored[config.TARGET] == 1
    flagged = scored["predicted_flag"] == 1
    kpis = pd.DataFrame([{
        "total_transactions": total,
        "total_fraud_transactions": int(fraud_rows.sum()),
        "fraud_rate_pct": float(100 * fraud_rows.mean()),
        "total_transaction_value": float(scored["amount"].sum()),
        "fraudulent_transaction_value": float(scored.loc[fraud_rows, "amount"].sum()),
        "selected_threshold": float(optimum["optimal_threshold"]),
        "fraud_capture_rate_pct": float(100 * optimum["fraud_capture_rate"]),
        "false_positive_rate_pct": float(100 * optimum["false_positive_rate"]),
        "precision_pct": float(100 * optimum["precision"]),
        "alerts_raised": int(flagged.sum()),
        "legitimate_transactions_blocked": int(optimum["FP"]),
        "fraud_missed": int(optimum["FN"]),
        "simulated_total_cost": float(optimum["total_cost_at_optimum"]),
        "simulated_cost_missed_fraud": float(optimum["cost_missed_fraud"]),
        "simulated_cost_customer_friction": float(optimum["cost_customer_friction"]),
        "simulated_cost_at_threshold_0_50": float(optimum["cost_at_threshold_0_50"]),
        "model_roc_auc": ranking.get("roc_auc"),
        "model_pr_auc": ranking.get("pr_auc"),
        "cost_assumption_false_negative": config.COST_FALSE_NEGATIVE,
        "cost_assumption_false_positive": config.COST_FALSE_POSITIVE,
        "cost_basis": "SIMULATED / ESTIMATED UNDER ASSUMPTIONS",
    }])
    written["kpi_summary"] = _write(kpis, "kpi_summary.csv", directory)

    # ---- curves and grids ------------------------------------------------
    written["threshold_cost_curve"] = _write(sweep, "threshold_cost_curve.csv", directory)
    written["cost_sensitivity"] = _write(sensitivity, "cost_sensitivity.csv", directory)

    # ---- fraud by type ---------------------------------------------------
    by_type = scored.groupby("type", observed=True).agg(
        transactions=("amount", "size"),
        fraud=(config.TARGET, "sum"),
        total_amount=("amount", "sum"),
        fraud_amount=("amount", lambda s: float(s[scored.loc[s.index, config.TARGET] == 1].sum())),
        alerts=("predicted_flag", "sum"),
        false_positives=("outcome_class", lambda s: int((s == "false_positive").sum())),
        true_positives=("outcome_class", lambda s: int((s == "true_positive").sum())),
        false_negatives=("outcome_class", lambda s: int((s == "false_negative").sum())),
    ).reset_index()
    by_type["fraud_rate_pct"] = 100 * by_type["fraud"] / by_type["transactions"]
    by_type["false_positive_rate_pct"] = (
        100 * by_type["false_positives"] / (by_type["transactions"] - by_type["fraud"]).replace(0, np.nan))
    written["fraud_by_type"] = _write(by_type, "fraud_by_type.csv", directory)

    # ---- trend over time -------------------------------------------------
    over_time = scored.groupby("step", observed=True).agg(
        transactions=("amount", "size"),
        fraud=(config.TARGET, "sum"),
        alerts=("predicted_flag", "sum"),
        total_amount=("amount", "sum"),
    ).reset_index()
    for outcome in ["true_positive", "false_positive", "false_negative"]:
        counts = (scored[scored["outcome_class"] == outcome]
                  .groupby("step", observed=True).size().rename(outcome))
        over_time = over_time.merge(counts, on="step", how="left")
    over_time = over_time.fillna(0)
    over_time["fraud_rate_pct"] = 100 * over_time["fraud"] / over_time["transactions"]
    written["fraud_over_time"] = _write(over_time, "fraud_over_time.csv", directory)

    # ---- false-positive profile -----------------------------------------
    legitimate = scored[scored[config.TARGET] == 0]
    profile = legitimate.groupby(
        ["type", "amount_band"], observed=True).agg(
        legitimate_transactions=("amount", "size"),
        false_positives=("outcome_class", lambda s: int((s == "false_positive").sum())),
        legitimate_value=("amount", "sum"),
    ).reset_index()
    profile["false_positive_rate_pct"] = (
        100 * profile["false_positives"] / profile["legitimate_transactions"])
    profile["simulated_friction_cost"] = profile["false_positives"] * config.COST_FALSE_POSITIVE
    written["false_positive_profile"] = _write(profile, "false_positive_profile.csv", directory)

    # ---- band definitions ------------------------------------------------
    band_table = pd.DataFrame(bands["bands"])
    band_table["derived_from"] = "cost-minimising thresholds for two intervention costs"
    counts = scored["risk_band"].value_counts()
    band_table["transactions_in_band"] = band_table["band"].map(counts).fillna(0).astype(int)
    fraud_in_band = scored[scored[config.TARGET] == 1]["risk_band"].value_counts()
    band_table["fraud_in_band"] = band_table["band"].map(fraud_in_band).fillna(0).astype(int)
    band_table["fraud_rate_in_band_pct"] = (
        100 * band_table["fraud_in_band"] / band_table["transactions_in_band"].replace(0, np.nan))
    written["risk_band_definition"] = _write(band_table, "risk_band_definition.csv", directory)

    return written
