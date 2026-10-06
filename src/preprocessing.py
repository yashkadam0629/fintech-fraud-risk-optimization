"""
Stage 0/1 loading layer.

The raw PaySim CSV is ~470 MB and 6.36 million rows. Read naively it costs
about 1.5 GB of RAM because every account name becomes a Python string.
Here it is read in chunks, compacted, and only then concatenated.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Money columns stay float64 on purpose: float32 carries only ~7 significant
# digits, and PaySim balances reach hundreds of millions with 2 decimals, so
# float32 would silently destroy the cents needed for balance reconciliation.
READ_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


class DatasetContractError(ValueError):
    """Raised when the CSV does not look like PaySim1."""


def find_dataset_file(directory: Path | None = None) -> Path:
    """Auto-detect the PaySim CSV instead of forcing an exact filename.

    Kaggle's file is PS_20174392719_1491204439457_log.csv, but people rename
    downloads all the time. We look for any CSV whose header matches the
    PaySim contract and pick the largest one.
    """
    directory = Path(directory) if directory else config.RAW_DIR
    candidates = sorted(directory.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {directory}.\n"
            "Download the dataset from https://www.kaggle.com/datasets/ealaxi/paysim1 "
            "and unzip the CSV into that folder. See data/README.md."
        )
    for path in candidates:
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        if all(col in header for col in config.EXPECTED_COLUMNS):
            logger.info("Detected dataset file: %s (%.0f MB)",
                        path.name, path.stat().st_size / 1024**2)
            return path
    raise DatasetContractError(
        f"CSV files were found in {directory} but none has the PaySim column set.\n"
        f"Expected columns: {config.EXPECTED_COLUMNS}"
    )


def validate_columns(columns: Iterable[str]) -> None:
    found = list(columns)
    missing = [c for c in config.EXPECTED_COLUMNS if c not in found]
    if missing:
        raise DatasetContractError(
            f"Missing expected PaySim columns: {missing}. Found: {found}")
    extra = [c for c in found if c not in config.EXPECTED_COLUMNS]
    if extra:
        logger.warning("Extra columns present and ignored: %s", extra)


def split_entity_id(names: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'C1231006815' -> ('C', 1231006815).

    Two reasons: memory (int64 is a fraction of the cost of a string), and
    meaning (the leading letter tells us customer vs merchant).
    """
    kind = names.str[0].astype("category")
    try:
        numeric = names.str[1:].astype("int64")
    except (ValueError, TypeError):
        logger.warning("Account IDs not in 'X<digits>' form; using factorize().")
        numeric = pd.Series(pd.factorize(names)[0], index=names.index, dtype="int64")
    return kind, numeric


def _compact(chunk: pd.DataFrame) -> pd.DataFrame:
    orig_kind, orig_id = split_entity_id(chunk["nameOrig"])
    dest_kind, dest_id = split_entity_id(chunk["nameDest"])
    return chunk.assign(
        orig_kind=orig_kind, orig_id=orig_id,
        dest_kind=dest_kind, dest_id=dest_id,
    ).drop(columns=["nameOrig", "nameDest"])


def load_paysim(path: Path | str | None = None, nrows: int | None = None,
                chunksize: int = 1_000_000) -> pd.DataFrame:
    """Load the dataset, sorted chronologically by `step`.

    `nrows` reads only the first N rows. PaySim is time-ordered, so that is an
    early time window, not a random sample. Useful for development runs only.
    """
    path = Path(path) if path is not None else find_dataset_file()
    validate_columns(pd.read_csv(path, nrows=0).columns)

    frames, total = [], 0
    for i, chunk in enumerate(pd.read_csv(path, dtype=READ_DTYPES,
                                          chunksize=chunksize, nrows=nrows), start=1):
        frames.append(_compact(chunk))
        total += len(chunk)
        logger.info("  chunk %d read, cumulative rows = %s", i, f"{total:,}")

    df = pd.concat(frames, ignore_index=True)
    del frames

    # Every behavioural feature downstream is a cumulative "past only"
    # calculation, which is only correct if rows are in chronological order.
    if not df["step"].is_monotonic_increasing:
        logger.info("Sorting by step to guarantee chronological order.")
        df = df.sort_values("step", kind="stable").reset_index(drop=True)

    logger.info("Loaded %s rows x %s cols (%.0f MB)", f"{len(df):,}", df.shape[1],
                df.memory_usage(deep=True).sum() / 1024**2)
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar-ish columns from `step`.

    PaySim has no timestamp. `step` is an hour counter from 1 to 744. So:
      sim_day     = day of the simulation (1-based)
      hour_of_day = position within the 24-hour cycle

    We do not know which real clock hour step 1 is, nor the calendar date, so
    weekday/weekend labels would be assumptions. `assumed_day_of_week` is
    provided for exploration only and is never used as a model feature.
    """
    step = df["step"].astype("int32")
    return df.assign(
        sim_day=((step - 1) // config.STEPS_PER_DAY + 1).astype("int16"),
        hour_of_day=((step - 1) % config.STEPS_PER_DAY).astype("int8"),
        assumed_day_of_week=(((step - 1) // config.STEPS_PER_DAY) % 7).astype("int8"),
    )


def add_balance_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostics for the leakage audit, not model features.

    Money-out:  newbalanceOrig should equal oldbalanceOrg - amount
    CASH_IN:    newbalanceOrig should equal oldbalanceOrg + amount
    Receiver:   newbalanceDest should equal oldbalanceDest + amount
    """
    sign = np.where(df["type"].astype(str) == "CASH_IN", 1.0, -1.0)
    return df.assign(
        balance_error_orig=df["newbalanceOrig"] - (df["oldbalanceOrg"] + sign * df["amount"]),
        balance_error_dest=df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"]),
    )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_balance_residuals(add_time_columns(df))
