"""
STAGE 1 - Data audit, leakage audit, honest baselines.

Every function answers one question a fraud analyst would actually ask and
returns a plain dictionary, so results can be serialised, rendered, or
asserted on in a test. Nothing here invents a column.
"""
from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src.utils import binary_metrics, to_native

logger = logging.getLogger(__name__)
TARGET = config.TARGET


# ==========================================================================
# structure & quality
# ==========================================================================
def profile_structure(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
    }


def profile_quality(df: pd.DataFrame) -> dict:
    """Missing values, duplicates, and impossible values.

    A duplicate row here is ambiguous: two identical transactions in the same
    simulated hour could be a genuine repeat payment. We count them and flag
    them for judgement; we do not silently delete them.
    """
    derived = {"sim_day", "hour_of_day", "assumed_day_of_week",
               "balance_error_orig", "balance_error_dest"}
    subset = [c for c in df.columns if c not in derived]
    dupes = int(df.duplicated(subset=subset).sum())

    issues = []
    if (df["amount"] < 0).any():
        issues.append("negative transaction amounts present")
    zero_amounts = int((df["amount"] == 0).sum())
    if zero_amounts:
        issues.append(f"{zero_amounts:,} transactions with amount = 0")
    if df["step"].min() < 1:
        issues.append("step values below 1")
    if dupes:
        issues.append(f"{dupes:,} exactly duplicated rows (may be genuine repeats)")

    return {
        "missing_values_per_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "total_missing_values": int(df.isna().sum().sum()),
        "exact_duplicate_rows": dupes,
        "duplicate_row_pct": float(100 * dupes / len(df)) if len(df) else 0.0,
        "zero_amount_transactions": zero_amounts,
        "negative_amount_transactions": int((df["amount"] < 0).sum()),
        "data_quality_issues": issues or ["no obvious structural problems detected"],
    }


# ==========================================================================
# step / time
# ==========================================================================
def profile_step(df: pd.DataFrame) -> dict:
    per_step = df.groupby("step", observed=True).size()
    per_day = df.groupby("sim_day", observed=True).agg(
        transactions=("amount", "size"), fraud=(TARGET, "sum"))
    hourly = df.groupby("hour_of_day", observed=True).agg(
        transactions=("amount", "size"), fraud=(TARGET, "sum"))
    hourly["fraud_rate_pct"] = 100 * hourly["fraud"] / hourly["transactions"]

    quiet = per_step.reindex(range(int(df["step"].min()), int(df["step"].max()) + 1)).fillna(0)
    quiet_hours = hourly.index[hourly["transactions"] < hourly["transactions"].median() * 0.25]

    return {
        "step_min": int(df["step"].min()),
        "step_max": int(df["step"].max()),
        "unique_steps": int(df["step"].nunique()),
        "missing_steps_in_range": int((quiet == 0).sum()),
        "implied_time_range": (
            f"{int(df['step'].max()) - int(df['step'].min()) + 1} hourly steps "
            f"= {(int(df['step'].max()) - int(df['step'].min()) + 1) / 24:.1f} days"
        ),
        "transactions_per_step_mean": float(per_step.mean()),
        "transactions_per_step_min": int(per_step.min()),
        "transactions_per_step_max": int(per_step.max()),
        "daily_volume_min": int(per_day["transactions"].min()),
        "daily_volume_max": int(per_day["transactions"].max()),
        "hourly_profile": to_native(hourly.reset_index().to_dict("records")),
        "daily_profile": to_native(per_day.reset_index().to_dict("records")),
        "low_volume_hours": [int(h) for h in quiet_hours],
        "fraud_rate_low_volume_hours_pct": float(
            100 * hourly.loc[quiet_hours, "fraud"].sum()
            / max(hourly.loc[quiet_hours, "transactions"].sum(), 1)),
        "fraud_rate_other_hours_pct": float(
            100 * hourly.drop(index=quiet_hours)["fraud"].sum()
            / max(hourly.drop(index=quiet_hours)["transactions"].sum(), 1)),
        "caveat": (
            "`step` is an hour index, not a timestamp. Sub-hourly features "
            "cannot be computed. Clock time and calendar date are unknown, so "
            "weekday/weekend labels would be assumptions, not facts."
        ),
    }


# ==========================================================================
# type / amount / target
# ==========================================================================
def profile_type(df: pd.DataFrame) -> dict:
    counts = df["type"].value_counts()
    by_type = df.groupby("type", observed=True).agg(
        transactions=("amount", "size"),
        fraud=(TARGET, "sum"),
        total_amount=("amount", "sum"),
        median_amount=("amount", "median"),
    )
    by_type["pct_of_transactions"] = 100 * by_type["transactions"] / len(df)
    by_type["fraud_rate_pct"] = 100 * by_type["fraud"] / by_type["transactions"]
    total_fraud = max(int(df[TARGET].sum()), 1)
    by_type["share_of_all_fraud_pct"] = 100 * by_type["fraud"] / total_fraud
    return {
        "unique_types": [str(t) for t in counts.index],
        "by_type": to_native(by_type.reset_index().to_dict("records")),
        "types_with_fraud": [str(t) for t in by_type.index[by_type["fraud"] > 0]],
        "types_without_fraud": [str(t) for t in by_type.index[by_type["fraud"] == 0]],
    }


def _amount_stats(series: pd.Series) -> dict:
    """Median and percentiles matter more than mean/std for skewed money data."""
    if len(series) == 0:
        return {"count": 0}
    q = series.quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_dict()
    return {
        "count": int(len(series)), "sum": float(series.sum()),
        "mean": float(series.mean()), "median": float(q[0.5]),
        "std": float(series.std()), "min": float(series.min()),
        "q1": float(q[0.25]), "q3": float(q[0.75]),
        "p90": float(q[0.9]), "p99": float(q[0.99]), "max": float(series.max()),
    }


def profile_amount(df: pd.DataFrame) -> dict:
    fraud = df[TARGET] == 1
    total_value = float(df["amount"].sum())
    fraud_value = float(df.loc[fraud, "amount"].sum())
    return {
        "all": _amount_stats(df["amount"]),
        "fraud": _amount_stats(df.loc[fraud, "amount"]),
        "legitimate": _amount_stats(df.loc[~fraud, "amount"]),
        "total_transaction_value": total_value,
        "fraudulent_transaction_value": fraud_value,
        "fraud_value_share_pct": float(100 * fraud_value / total_value) if total_value else 0.0,
    }


def profile_target(df: pd.DataFrame) -> dict:
    fraud = int(df[TARGET].sum())
    legit = int(len(df) - fraud)
    rate = fraud / len(df) if len(df) else 0.0
    return {
        "fraud_count": fraud,
        "legitimate_count": legit,
        "fraud_rate_pct": float(100 * rate),
        "imbalance_ratio_legit_per_fraud": float(legit / fraud) if fraud else None,
        "accuracy_of_always_legitimate_pct": float(100 * (1 - rate)),
    }


def profile_flagged(df: pd.DataFrame) -> dict:
    """isFlaggedFraud is the simulator's own control rule, not a model."""
    flagged = df["isFlaggedFraud"] == 1
    fraud = df[TARGET] == 1
    n_flagged = int(flagged.sum())
    return {
        "flagged_count": n_flagged,
        "flagged_pct": float(100 * n_flagged / len(df)),
        "flagged_and_actually_fraud": int((flagged & fraud).sum()),
        "flagged_and_legitimate": int((flagged & ~fraud).sum()),
        "fraud_not_flagged": int((~flagged & fraud).sum()),
        "share_of_fraud_flagged_pct": float(100 * (flagged & fraud).sum() / max(int(fraud.sum()), 1)),
        "types_flagged": [str(t) for t in df.loc[flagged, "type"].unique()],
        "min_flagged_amount": float(df.loc[flagged, "amount"].min()) if n_flagged else None,
    }


# ==========================================================================
# account repetition - decides whether customer history features are viable
# ==========================================================================
def _repetition_stats(ids: np.ndarray) -> dict:
    _, counts = np.unique(ids, return_counts=True)
    n_entities = int(counts.size)
    repeat_mask = counts > 1
    return {
        "unique_accounts": n_entities,
        "accounts_with_more_than_one_txn": int(repeat_mask.sum()),
        "pct_accounts_repeat": float(100 * repeat_mask.sum() / n_entities),
        "pct_transactions_from_repeat_accounts": float(
            100 * counts[repeat_mask].sum() / counts.sum()),
        "mean_txns_per_account": float(counts.mean()),
        "median_txns_per_account": float(np.median(counts)),
        "max_txns_from_one_account": int(counts.max()),
        "distribution_txns_per_account": {
            "1": int((counts == 1).sum()),
            "2": int((counts == 2).sum()),
            "3_to_5": int(((counts >= 3) & (counts <= 5)).sum()),
            "6_to_20": int(((counts >= 6) & (counts <= 20)).sum()),
            "21_plus": int((counts > 20).sum()),
        },
    }


def profile_accounts(df: pd.DataFrame) -> dict:
    """Measure, do not assume, whether accounts repeat.

    This decides the entire Stage 2 design. If originating accounts are
    effectively single-use, per-customer baselines are undefined for most rows
    and behavioural analysis must shift to the destination side.
    """
    origin = _repetition_stats(df["orig_id"].to_numpy())
    dest = _repetition_stats(df["dest_id"].to_numpy())

    # Fan-in: how many distinct senders has each destination received from.
    pair_first = ~df.duplicated(subset=["orig_id", "dest_id"])
    fan_in = df.loc[pair_first].groupby("dest_id", observed=True).size()
    dest["fan_in_mean_unique_senders"] = float(fan_in.mean())
    dest["fan_in_max_unique_senders"] = int(fan_in.max())
    dest["fan_in_p99_unique_senders"] = float(fan_in.quantile(0.99))

    prior_orig = df.groupby("orig_id", sort=False).cumcount()
    prior_dest = df.groupby("dest_id", sort=False).cumcount()
    fraud = df[TARGET] == 1

    counterparty = df.groupby("dest_kind", observed=True).agg(
        transactions=("amount", "size"), fraud=(TARGET, "sum"))
    counterparty["fraud_rate_pct"] = 100 * counterparty["fraud"] / counterparty["transactions"]

    return {
        "origin": origin,
        "destination": dest,
        "prior_history_coverage": {
            "rows_with_prior_origin_history_pct": float(100 * (prior_orig > 0).mean()),
            "rows_with_prior_destination_history_pct": float(100 * (prior_dest > 0).mean()),
            "fraud_rows_with_prior_origin_history_pct": float(
                100 * (prior_orig[fraud] > 0).mean()) if fraud.any() else None,
            "fraud_rows_with_prior_destination_history_pct": float(
                100 * (prior_dest[fraud] > 0).mean()) if fraud.any() else None,
            "interpretation": (
                "These percentages are the hard ceiling on coverage for any "
                "per-account historical feature. A feature defined for 3% of "
                "rows cannot carry a model."
            ),
        },
        "counterparty_kind": to_native(counterparty.reset_index().to_dict("records")),
        "design_decision": _account_design_decision(origin, dest),
    }


def _account_design_decision(origin: dict, dest: dict) -> str:
    if origin["pct_transactions_from_repeat_accounts"] >= 30:
        return ("Originating accounts repeat often enough to support "
                "customer-level historical features; build both origin and "
                "destination behaviour.")
    return ("Originating accounts are largely single-use, so per-customer "
            "historical baselines would be undefined for most transactions. "
            "Behavioural analysis is shifted to destination behaviour, "
            "sender-destination relationships, counterparty fan-in and "
            "time-window velocity. Origin features are still computed and "
            "reported, with their coverage stated, rather than fabricated.")


# ==========================================================================
# leakage audit
# ==========================================================================
def leakage_audit(df: pd.DataFrame) -> dict:
    """Quantify how much each suspect column gives the answer away.

    The dataset documentation states that fraudulent transactions are
    cancelled in the simulation, so the balance fields are recorded after that
    reversal. We measure the size of the problem rather than assume it.
    """
    tol = config.BALANCE_TOLERANCE
    fraud = df[TARGET] == 1

    def share(mask: pd.Series, subset: pd.Series) -> float:
        return float(100 * mask[subset].mean()) if subset.any() else 0.0

    checks = {
        "origin_balance_does_not_reconcile": df["balance_error_orig"].abs() > tol,
        "destination_balance_does_not_reconcile": df["balance_error_dest"].abs() > tol,
        "amount_equals_entire_origin_balance": (df["oldbalanceOrg"] - df["amount"]).abs() < tol,
        "origin_balance_emptied_to_zero": df["newbalanceOrig"].abs() < tol,
        "destination_balances_both_zero": (
            (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)),
    }
    measured = {
        name: {"fraud_pct": share(mask, fraud),
               "legitimate_pct": share(mask, ~fraud),
               "separation_pct_points": share(mask, fraud) - share(mask, ~fraud)}
        for name, mask in checks.items()
    }

    # Single-feature discriminative power of each suspect column.
    y = df[TARGET].to_numpy()
    univariate = {}
    for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
                "newbalanceDest", "balance_error_orig", "balance_error_dest",
                "isFlaggedFraud", "amount"]:
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype="float64")
        if 0 < y.sum() < len(y):
            univariate[col] = {
                "roc_auc_single_feature": float(roc_auc_score(y, values)),
                "pr_auc_single_feature": float(average_precision_score(y, values)),
            }

    decisions = [
    {"column": "oldbalanceOrg", "leakage_risk": "MEDIUM",
     "reason": ("Pre-transaction origin balance. It is available before the "
                "transaction, but is excluded from the primary model "
                "conservatively because PaySim's balance and reversal mechanics "
                "can create simulator-specific behavioural signals."),
     "primary_model_decision": "EXCLUDE"},
    {"column": "newbalanceOrig", "leakage_risk": "HIGH",
     "reason": ("Post-transaction origin balance. It describes the account "
                "after the transaction and is therefore not available when "
                "the authorization decision is made."),
     "primary_model_decision": "EXCLUDE"},
    {"column": "oldbalanceDest", "leakage_risk": "MEDIUM",
     "reason": ("Pre-transaction destination balance. It is available before "
                "the transaction, but is excluded from the primary model "
                "conservatively because of PaySim's simulator-specific "
                "balance and reversal mechanics."),
     "primary_model_decision": "EXCLUDE"},
    {"column": "newbalanceDest", "leakage_risk": "HIGH",
     "reason": ("Post-transaction destination balance. It describes the "
                "destination account after the transaction and is therefore "
                "not available at authorization time."),
     "primary_model_decision": "EXCLUDE"},
    {"column": "isFlaggedFraud", "leakage_risk": "HIGH",
     "reason": ("A control output, not an independent behavioural observation. "
                "It is excluded from the primary model because it represents "
                "a simulator-generated fraud flag rather than a clean input "
                "available for independent modelling."),
     "primary_model_decision": "EXCLUDE (kept as a benchmark baseline)"},
    {"column": "amount", "leakage_risk": "LOW",
     "reason": "Known before the decision is made; a genuine input.",
     "primary_model_decision": "INCLUDE"},
    {"column": "type", "leakage_risk": "LOW",
     "reason": "Known at request time.",
     "primary_model_decision": "INCLUDE"},
    {"column": "step", "leakage_risk": "LOW (use with care)",
     "reason": ("Known at request time, but the raw index encodes position "
                "in the simulation and does not generalise forward. Only "
                "hour-of-day is used as a feature."),
     "primary_model_decision": "INCLUDE as hour_of_day only"},
]

    return {
        "balance_pattern_checks_pct": measured,
        "single_feature_discrimination": univariate,
        "decision_table": decisions,
        "conclusion": (
            "Balance columns are excluded from the primary model. A separate, "
            "clearly labelled leakage demonstration model may be trained on "
            "them to show how much performance they artificially add."
        ),
    }


# ==========================================================================
# honest baselines
# ==========================================================================
def honest_baselines(df: pd.DataFrame) -> dict:
    """Reference points computed before any model exists.

    ROC-AUC and PR-AUC need a continuous score, which is why they appear for
    the amount ranking but not for the binary rules.
    """
    y = df[TARGET].to_numpy()
    prevalence = float(y.mean())

    rules = {
        "baseline_1_flag_nothing": np.zeros_like(y),
        "baseline_2_flag_everything": np.ones_like(y),
        "baseline_3_isFlaggedFraud": df["isFlaggedFraud"].to_numpy(),
        "baseline_4_transfer_or_cashout":
            df["type"].astype(str).isin(["TRANSFER", "CASH_OUT"]).to_numpy().astype(int),
    }
    results = {name: binary_metrics(y, pred) for name, pred in rules.items()}

    amount = df["amount"].to_numpy()
    if 0 < y.sum() < len(y):
        roc = float(roc_auc_score(y, amount))
        pr = float(average_precision_score(y, amount))
    else:
        roc = pr = None

    # Amount-only, thresholded at its 99th percentile, so it can also be read
    # as a deployable rule rather than only as a ranking.
    cutoff = float(np.quantile(amount, 0.99))
    results["baseline_5_amount_above_p99"] = binary_metrics(y, (amount >= cutoff).astype(int))

    return {
        "fraud_prevalence": prevalence,
        "rules": results,
        "baseline_5_amount_ranking": {
            "roc_auc": roc,
            "pr_auc": pr,
            "pr_auc_no_skill_baseline": prevalence,
            "pr_auc_lift_over_no_skill": float(pr / prevalence) if (pr and prevalence) else None,
            "p99_amount_cutoff": cutoff,
        },
        "why_accuracy_misleads": (
            f"Predicting 'legitimate' for every transaction scores "
            f"{100 * results['baseline_1_flag_nothing']['accuracy']:.3f}% accuracy "
            f"and catches zero fraud. Accuracy is dominated by the majority "
            f"class and cannot distinguish a useful fraud model from a useless one."
        ),
        "why_pr_auc_matters": (
            "ROC-AUC's false-positive rate has the huge legitimate population "
            "in its denominator, so it stays flattering even when precision is "
            "poor. PR-AUC uses precision, whose denominator is the small set of "
            "flagged transactions, so it responds directly to alert quality. "
            f"Its no-skill baseline equals the fraud rate ({100 * prevalence:.4f}%)."
        ),
        "benchmark_note": (
            "Stage 3 models must beat these numbers on the same evaluation "
            "sample to justify their existence."
        ),
    }


# ==========================================================================
# orchestration
# ==========================================================================
def run_full_audit(df: pd.DataFrame) -> "OrderedDict[str, dict]":
    sections = OrderedDict()
    steps = [
        ("structure", profile_structure),
        ("data_quality", profile_quality),
        ("step_and_time", profile_step),
        ("transaction_type", profile_type),
        ("amount", profile_amount),
        ("target_isFraud", profile_target),
        ("isFlaggedFraud", profile_flagged),
        ("account_repetition", profile_accounts),
        ("leakage_audit", leakage_audit),
        ("honest_baselines", honest_baselines),
    ]
    for name, func in steps:
        logger.info("  audit section: %s", name)
        try:
            sections[name] = to_native(func(df))
        except Exception as exc:
            logger.exception("Audit section '%s' failed", name)
            sections[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return sections
