"""
Tests for the invariants that actually matter.

The point of these is not coverage. It is that the three mistakes which would
silently invalidate the whole project each have a test:

  1. a feature that peeks at the present or the future
  2. a split that leaks time
  3. a cost curve that does not actually minimise cost

Run:  pytest tests/ -q        (pip install pytest)
Or:   python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, data_profiling, feature_engineering, preprocessing
from src import threshold_optimization, train_models
from tests.make_sample_data import make_sample


def _prepared(rows: int = 20_000) -> pd.DataFrame:
    frame = make_sample(rows, seed=7)
    kind_o, id_o = preprocessing.split_entity_id(frame["nameOrig"].astype("string"))
    kind_d, id_d = preprocessing.split_entity_id(frame["nameDest"].astype("string"))
    frame = (frame.assign(orig_kind=kind_o, orig_id=id_o, dest_kind=kind_d, dest_id=id_d)
             .drop(columns=["nameOrig", "nameDest"]))
    frame["type"] = frame["type"].astype("category")
    frame = frame.sort_values("step", kind="stable").reset_index(drop=True)
    return preprocessing.prepare(frame)


# ==========================================================================
# contract
# ==========================================================================
def test_column_contract_rejects_wrong_schema():
    try:
        preprocessing.validate_columns(["step", "amount"])
    except preprocessing.DatasetContractError:
        return
    raise AssertionError("validate_columns should reject an incomplete schema")


def test_time_columns_in_range():
    df = _prepared()
    assert df["hour_of_day"].between(0, 23).all()
    assert df["sim_day"].min() >= 1


# ==========================================================================
# leakage safety - the tests that matter most
# ==========================================================================
def test_first_transaction_of_an_account_has_no_prior_history():
    """If a frame ended at CURRENT ROW, the first row would already have history."""
    df = _prepared()
    featured, _ = feature_engineering.build_features(df)
    for id_col, count_col in (("orig_id", "orig_prior_txn_count"),
                              ("dest_id", "dest_prior_txn_count")):
        first = featured.groupby(id_col, sort=False).cumcount() == 0
        assert (featured.loc[first, count_col] == 0).all(), f"{count_col} leaks the current row"
        assert featured.loc[first, count_col.replace("txn_count", "avg_amount")].isna().all()


def test_velocity_windows_exclude_the_current_hour():
    """A destination's very first transaction must show zero prior velocity."""
    df = _prepared()
    featured, _ = feature_engineering.build_features(df)
    first = featured.groupby("dest_id", sort=False).cumcount() == 0
    for window in config.VELOCITY_WINDOWS_HOURS:
        assert (featured.loc[first, f"dest_txn_prev_{window}h"] == 0).all()


def test_velocity_windows_are_nested():
    """24h counts can never exceed 72h counts, which can never exceed 168h."""
    df = _prepared()
    featured, _ = feature_engineering.build_features(df)
    assert (featured["dest_txn_prev_24h"] <= featured["dest_txn_prev_72h"]).all()
    assert (featured["dest_txn_prev_72h"] <= featured["dest_txn_prev_168h"]).all()


def test_prior_window_counts_matches_a_brute_force_reference():
    """The searchsorted implementation must agree with the obvious slow version."""
    rng = np.random.default_rng(0)
    entity = rng.integers(0, 12, size=400)
    step = np.sort(rng.integers(1, 200, size=400))
    window = 24

    fast = feature_engineering.prior_window_counts(entity, step, window)
    slow = np.array([
        int(((entity == entity[i]) & (step >= step[i] - window) & (step < step[i])).sum())
        for i in range(len(entity))
    ])
    assert np.array_equal(fast, slow)


def test_fan_in_never_exceeds_transaction_count():
    """Distinct senders cannot outnumber transactions received."""
    df = _prepared()
    featured, _ = feature_engineering.build_features(df)
    assert (featured["dest_prior_unique_senders"] <= featured["dest_prior_txn_count"]).all()


def test_leakage_columns_are_not_in_the_feature_set():
    df = _prepared()
    _, feature_columns = feature_engineering.build_features(df)
    for column in config.LEAKAGE_SUSPECT_COLUMNS:
        assert column not in feature_columns, f"{column} must not be a model feature"
    assert "step" not in feature_columns, "raw step does not generalise forward"
    assert config.TARGET not in feature_columns


# ==========================================================================
# split
# ==========================================================================
def test_time_split_is_chronological_and_disjoint():
    df = _prepared()
    featured, _ = feature_engineering.build_features(df)
    masks = train_models.time_aware_split(featured)["masks"]

    train_max = featured.loc[masks["train"], "step"].max()
    val_min = featured.loc[masks["validation"], "step"].min()
    val_max = featured.loc[masks["validation"], "step"].max()
    test_min = featured.loc[masks["test"], "step"].min()

    assert train_max < val_min, "train and validation overlap in time"
    assert val_max < test_min, "validation and test overlap in time"
    total = sum(int(m.sum()) for m in masks.values())
    assert total == len(featured), "splits do not partition the data"


# ==========================================================================
# cost model
# ==========================================================================
def test_sweep_confusion_counts_are_consistent():
    rng = np.random.default_rng(1)
    y = (rng.random(5_000) < 0.02).astype(int)
    scores = np.clip(rng.random(5_000) * 0.5 + y * 0.3, 0, 1)
    sweep = threshold_optimization.threshold_sweep(y, scores)

    assert (sweep[["TP", "TN", "FP", "FN"]].sum(axis=1) == len(y)).all()
    assert (sweep["TP"] + sweep["FN"] == y.sum()).all()
    # Raising the threshold can never increase the number of alerts.
    assert sweep.sort_values("threshold")["alert_volume"].is_monotonic_decreasing


def test_optimal_threshold_really_is_the_minimum():
    rng = np.random.default_rng(2)
    y = (rng.random(5_000) < 0.02).astype(int)
    scores = np.clip(rng.random(5_000) * 0.5 + y * 0.3, 0, 1)
    sweep = threshold_optimization.threshold_sweep(y, scores)
    best = threshold_optimization.optimal_threshold(sweep)
    assert best["total_cost_at_optimum"] == sweep["total_cost"].min()


def test_cheaper_intervention_implies_a_lower_threshold():
    """The MEDIUM band only exists if the verify threshold sits below block."""
    rng = np.random.default_rng(3)
    y = (rng.random(8_000) < 0.03).astype(int)
    scores = np.clip(rng.random(8_000) * 0.4 + y * 0.4, 0, 1)
    bands = threshold_optimization.derive_risk_bands(y, scores)
    assert bands["medium_risk_threshold_verify"] <= bands["high_risk_threshold_block"]


# ==========================================================================
# audit
# ==========================================================================
def test_audit_sections_all_run():
    audit = data_profiling.run_full_audit(_prepared())
    expected = {"structure", "data_quality", "step_and_time", "transaction_type",
                "amount", "target_isFraud", "isFlaggedFraud", "account_repetition",
                "leakage_audit", "honest_baselines"}
    assert expected.issubset(audit.keys())
    assert all("error" not in section for section in audit.values())


def test_flag_nothing_accuracy_equals_one_minus_prevalence():
    audit = data_profiling.run_full_audit(_prepared())
    baselines = audit["honest_baselines"]
    accuracy = baselines["rules"]["baseline_1_flag_nothing"]["accuracy"]
    assert abs(accuracy - (1 - baselines["fraud_prevalence"])) < 1e-9


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"ok    {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print("all checks passed" if not failures else f"{failures} failing test(s)")
    raise SystemExit(1 if failures else 0)
