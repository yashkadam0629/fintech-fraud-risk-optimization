"""
STAGE 2 - Behavioural feature engineering (Python implementation).

This mirrors sql/03_behavioral_features.sql one-for-one. The SQL is the
portfolio artifact showing window-function fluency; this module is what the
model actually trains on, so the two must stay in agreement.

THE ONE RULE THAT MATTERS
Every historical feature uses information strictly BEFORE the current
transaction. Concretely:
  - expanding aggregates are shifted so the current row is excluded
  - time windows cover earlier `step` values only, never the current hour
  - nothing is computed from the target

WHAT IS DELIBERATELY NOT BUILT
  - minute or second level velocity: `step` is hourly, so it does not exist
  - any geographic feature: PaySim has no location data at all
  - a true 30-day rolling baseline: the dataset is only ~30 days long, so an
    expanding "everything known so far" baseline is used instead, and its
    coverage is reported rather than assumed
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# `step` never exceeds 744, so 1000 is a safe radix for packing
# (entity_id, step) into one sortable int64 key.
_STEP_RADIX = 1000


def prior_window_counts(entity_id: np.ndarray, step: np.ndarray, window_hours: int) -> np.ndarray:
    """Count earlier transactions for the same entity within a time window.

    Returns, for each row, the number of transactions by the same entity whose
    step lies in [step - window_hours, step - 1]. The current hour is excluded,
    which is what makes the feature leakage-safe.

    This is the vectorised equivalent of the SQL:
        COUNT(*) OVER (PARTITION BY entity ORDER BY step
                       RANGE BETWEEN <window> PRECEDING AND 1 PRECEDING)

    Implementation: sort by (entity, step), pack both into one monotonically
    increasing int64 key, and use two binary searches per row. O(n log n) for
    6 million rows instead of a Python loop.
    """
    entity_id = entity_id.astype(np.int64)
    step = step.astype(np.int64)

    order = np.lexsort((step, entity_id))
    sorted_entity = entity_id[order]
    sorted_step = step[order]
    keys = sorted_entity * _STEP_RADIX + sorted_step

    upper = keys                                        # first row at this entity+step
    lower = sorted_entity * _STEP_RADIX + np.maximum(sorted_step - window_hours, 0)

    counts_sorted = (np.searchsorted(keys, upper, side="left")
                     - np.searchsorted(keys, lower, side="left"))

    out = np.empty_like(counts_sorted)
    out[order] = counts_sorted
    return out.astype(np.int32)


def _prior_expanding(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Expanding past-only aggregates for one entity column.

    SQL equivalent:
        COUNT(*)  OVER (PARTITION BY key ORDER BY step
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        AVG(amount) OVER (... same frame ...)
        MAX(amount) OVER (... same frame ...)
    """
    grouped = df.groupby(key, sort=False)

    prior_count = grouped.cumcount().astype("int32")
    prior_sum = grouped["amount"].cumsum() - df["amount"]
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_avg = prior_sum / prior_count.where(prior_count > 0)

    # cummax over the *previous* amount gives the maximum excluding the
    # current row (groupby.cummax skips NaN, so the first row stays NaN).
    previous_amount = grouped["amount"].shift(1)
    prior_max = previous_amount.groupby(df[key], sort=False).cummax()

    steps_since_previous = df["step"] - grouped["step"].shift(1)

    return pd.DataFrame({
        f"{prefix}_prior_txn_count": prior_count,
        f"{prefix}_prior_total_amount": prior_sum.fillna(0.0),
        f"{prefix}_prior_avg_amount": prior_avg,
        f"{prefix}_prior_max_amount": prior_max,
        f"{prefix}_steps_since_prev_txn": steps_since_previous,
        f"{prefix}_has_history": (prior_count > 0).astype("int8"),
    }, index=df.index)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the dataframe with behavioural features plus the feature list.

    Assumes `df` is already sorted chronologically by `step` (preprocessing
    guarantees this). Order is not cosmetic here: every cumulative feature
    below is wrong if the rows are shuffled.
    """
    if not df["step"].is_monotonic_increasing:
        raise ValueError("Rows must be sorted by step before feature engineering.")

    out = df.copy()

    # ---- amount shape ---------------------------------------------------
    out["log_amount"] = np.log10(out["amount"].clip(lower=0) + 1.0)
    # Round-number amounts are a commonly cited manual-entry / scripted signal.
    out["amount_is_round_1000"] = (out["amount"] % 1000 == 0).astype("int8")

    # ---- origin behaviour ----------------------------------------------
    logger.info("  origin expanding history")
    origin = _prior_expanding(out, "orig_id", "orig")
    out = pd.concat([out, origin], axis=1)
    out["orig_amount_to_prior_avg"] = (
        out["amount"] / out["orig_prior_avg_amount"]).replace([np.inf, -np.inf], np.nan)
    out["orig_amount_minus_prior_avg"] = out["amount"] - out["orig_prior_avg_amount"]
    out["orig_amount_to_prior_max"] = (
        out["amount"] / out["orig_prior_max_amount"]).replace([np.inf, -np.inf], np.nan)

    # ---- destination behaviour -----------------------------------------
    logger.info("  destination expanding history")
    dest = _prior_expanding(out, "dest_id", "dest")
    out = pd.concat([out, dest], axis=1)
    out["dest_amount_to_prior_avg"] = (
        out["amount"] / out["dest_prior_avg_amount"]).replace([np.inf, -np.inf], np.nan)
    out["dest_is_new"] = (out["dest_prior_txn_count"] == 0).astype("int8")
    out["dest_is_merchant"] = (out["dest_kind"].astype(str) == "M").astype("int8")

    # ---- sender-destination relationship --------------------------------
    logger.info("  sender-destination relationship")
    pair = out.groupby(["orig_id", "dest_id"], sort=False)
    out["pair_prior_txn_count"] = pair.cumcount().astype("int32")
    out["pair_is_new"] = (out["pair_prior_txn_count"] == 0).astype("int8")
    out["pair_prior_total_amount"] = (pair["amount"].cumsum() - out["amount"]).fillna(0.0)

    # Destination fan-in: number of DISTINCT senders seen before this row.
    # Counting a new sender exactly once, then taking a past-only cumulative
    # sum within the destination, gives an exact distinct count with no loop.
    is_first_contact = (out["pair_prior_txn_count"] == 0).astype("int32")
    cumulative_distinct = is_first_contact.groupby(out["dest_id"], sort=False).cumsum()
    out["dest_prior_unique_senders"] = (cumulative_distinct - is_first_contact).astype("int32")

    # ---- velocity over genuine time windows ------------------------------
    logger.info("  time-window velocity (RANGE-equivalent)")
    step_values = out["step"].to_numpy()
    for window in config.VELOCITY_WINDOWS_HOURS:
        out[f"dest_txn_prev_{window}h"] = prior_window_counts(
            out["dest_id"].to_numpy(), step_values, window)
    out["orig_txn_prev_24h"] = prior_window_counts(
        out["orig_id"].to_numpy(), step_values, 24)

    # Same-hour concurrency. This uses transactions inside the current step,
    # which is the finest resolution the data offers. It is concurrent
    # information, not future information, and is flagged as such.
    out["dest_txn_prev_1h"] = prior_window_counts(
        out["dest_id"].to_numpy(),
        step_values,
        1
    )

    # Velocity acceleration: recent rate versus the longer-run rate.
    long_window = max(config.VELOCITY_WINDOWS_HOURS)
    out["dest_velocity_ratio_24h_vs_168h"] = (
        out["dest_txn_prev_24h"]
        / (out[f"dest_txn_prev_{long_window}h"].replace(0, np.nan) / (long_window / 24))
    ).replace([np.inf, -np.inf], np.nan)

    # ---- transaction type baseline ---------------------------------------
    # Expanding average amount for the type, past-only, so "is this large for
    # its own category" is answerable without peeking ahead.
    type_group = out.groupby("type", observed=True, sort=False)
    type_prior_count = type_group.cumcount()
    type_prior_sum = type_group["amount"].cumsum() - out["amount"]
    out["type_prior_avg_amount"] = type_prior_sum / type_prior_count.where(type_prior_count > 0)
    out["amount_to_type_prior_avg"] = (
        out["amount"] / out["type_prior_avg_amount"]).replace([np.inf, -np.inf], np.nan)

    # ---- one-hot transaction type ----------------------------------------
    type_dummies = pd.get_dummies(out["type"].astype(str), prefix="type").astype("int8")
    out = pd.concat([out, type_dummies], axis=1)

    feature_columns = [
        # amount
        "amount", "log_amount", "amount_is_round_1000",
        "amount_to_type_prior_avg",
        # time of day (step itself is excluded: it does not generalise forward)
        "hour_of_day",
        # origin behaviour
        "orig_prior_txn_count", "orig_prior_avg_amount", "orig_prior_max_amount",
        "orig_prior_total_amount", "orig_steps_since_prev_txn", "orig_has_history",
        "orig_amount_to_prior_avg", "orig_amount_minus_prior_avg",
        "orig_amount_to_prior_max", "orig_txn_prev_24h",
        # destination behaviour
        "dest_prior_txn_count", "dest_prior_avg_amount", "dest_prior_max_amount",
        "dest_prior_total_amount", "dest_steps_since_prev_txn", "dest_has_history",
        "dest_amount_to_prior_avg", "dest_is_new", "dest_is_merchant",
        "dest_prior_unique_senders",
        # velocity
        *[f"dest_txn_prev_{w}h" for w in config.VELOCITY_WINDOWS_HOURS],
        "dest_txn_prev_1h", "dest_velocity_ratio_24h_vs_168h",
        # relationship
        "pair_prior_txn_count", "pair_is_new", "pair_prior_total_amount",
        # type
        *sorted(type_dummies.columns),
    ]
    feature_columns = [c for c in feature_columns if c in out.columns]

    logger.info("  built %d features", len(feature_columns))
    return out, feature_columns


def feature_coverage(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """How often is each feature actually defined?

    A feature that is NaN for 97% of rows is not a feature, it is a footnote.
    This table is what stops us from shipping fake customer-history signals.
    """
    rows = []
    for col in feature_columns:
        series = df[col]
        rows.append({
            "feature": col,
            "defined_pct": float(100 * series.notna().mean()),
            "nonzero_pct": float(100 * (series.fillna(0) != 0).mean()),
            "mean": float(pd.to_numeric(series, errors="coerce").mean()),
            "median": float(pd.to_numeric(series, errors="coerce").median()),
        })
    return pd.DataFrame(rows).sort_values("defined_pct")


def fill_missing(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Impute undefined history with 0, paired with *_has_history flags.

    Zero is not a neutral value here, it is a statement: "no prior history".
    The has_history indicator columns let a model separate "no history" from
    "history whose value happens to be zero", which is why they exist.
    """
    filled = df.copy()
    filled[feature_columns] = (
        filled[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )
    return filled
