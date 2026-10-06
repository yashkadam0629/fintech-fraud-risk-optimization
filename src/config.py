"""
Single source of truth for paths, dataset contract, and every business
assumption. No analytical module hard-codes a constant.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_RESULT_DIR = OUTPUT_DIR / "model_results"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard_data"
REPORT_DIR = PROJECT_ROOT / "reports"

ALL_OUTPUT_DIRS = [PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR, MODEL_RESULT_DIR,
                   DASHBOARD_DIR, REPORT_DIR]

# --------------------------------------------------------------------------
# Dataset contract (PaySim1, Kaggle: ealaxi/paysim1)
# --------------------------------------------------------------------------
# Note the inconsistent spelling in the source file: "oldbalanceOrg" has no
# 'i', "newbalanceOrig" does. That is in the original data, not a typo here.
EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
TARGET = "isFraud"

# Columns excluded from the primary model. See the leakage audit in
# src/data_profiling.py for the measured justification.
LEAKAGE_SUSPECT_COLUMNS = [
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud",
]

STEPS_PER_DAY = 24
MAX_STEP_EXPECTED = 744          # 30-day simulation, hourly steps
BALANCE_TOLERANCE = 0.01

# Behavioural time windows, expressed in steps (= hours). The dataset has no
# sub-hourly resolution, so minute-level windows are not defined here and must
# never be added.
VELOCITY_WINDOWS_HOURS = [24, 72, 168]

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Time-aware split (fractions of transaction volume, ordered by step)
# --------------------------------------------------------------------------
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20   # test gets the remaining 0.20

# --------------------------------------------------------------------------
# SIMULATED business cost assumptions
# These are ASSUMPTIONS, not measured figures from any company.
# --------------------------------------------------------------------------
COST_FALSE_NEGATIVE = 500.0   # one missed fraudulent transaction
COST_FALSE_POSITIVE = 25.0    # one blocked legitimate transaction
COST_STEP_UP_VERIFICATION = 5.0  # softer intervention, used to derive the
                                 # MEDIUM risk band boundary

SENSITIVITY_FN_COSTS = [100.0, 250.0, 500.0, 750.0, 1000.0]
SENSITIVITY_FP_COSTS = [5.0, 10.0, 25.0, 50.0, 100.0]

THRESHOLD_GRID_START = 0.01
THRESHOLD_GRID_STOP = 0.99
THRESHOLD_GRID_STEP = 0.01

# The fixed 0.01 grid can be too coarse when a calibrated model puts almost all
# of its mass below 0.05. We therefore ALSO evaluate thresholds at quantiles of
# the score distribution, so the reported minimum is not an artifact of grid
# spacing. Both sets appear in the sweep, tagged in the `grid_source` column.
QUANTILE_GRID_POINTS = 80
QUANTILE_GRID_MIN = 0.90      # focus on the top of the score distribution

# --------------------------------------------------------------------------
# Model settings
# --------------------------------------------------------------------------
# Class weighting is OFF by default, deliberately.
#
# class_weight="balanced" makes a model output inflated probabilities: a score
# of 0.9 no longer means "90% likely fraud". That destroys the calibration the
# whole cost model depends on, and it pushes every cost-minimising threshold up
# against 0.99 where the grid can barely resolve it.
#
# This project handles imbalance at the DECISION layer instead, which is the
# entire point of Stage 3B: the threshold is what trades recall against
# precision, and it does so with an explicit price attached. Pass
# --class-weight balanced to compare.
CLASS_WEIGHT = None

LOGREG_PARAMS = dict(max_iter=1000, class_weight=CLASS_WEIGHT,
                     random_state=RANDOM_SEED)

RANDOM_FOREST_PARAMS = dict(
    n_estimators=120, max_depth=14, min_samples_leaf=40,
    max_features="sqrt", class_weight=CLASS_WEIGHT,
    n_jobs=-1, random_state=RANDOM_SEED,
)

# Reduced settings used with --fast, for laptops or a quick first pass.
RANDOM_FOREST_PARAMS_FAST = dict(
    n_estimators=40, max_depth=10, min_samples_leaf=100,
    max_features="sqrt", class_weight=CLASS_WEIGHT,
    max_samples=0.3, n_jobs=-1, random_state=RANDOM_SEED,
)

XGBOOST_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
    random_state=RANDOM_SEED, n_jobs=-1,
)


def ensure_directories() -> None:
    for directory in ALL_OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
