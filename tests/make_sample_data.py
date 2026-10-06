"""
Generate a CSV with the same SHAPE as PaySim1 so the pipeline can be
smoke-tested in seconds without the 470 MB download.

This is a plumbing test only. The numbers it produces are invented and must
never appear in any report, README, or conclusion. Real analysis runs against
the real Kaggle file.

    python tests/make_sample_data.py --rows 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
TYPE_WEIGHTS = [0.34, 0.08, 0.35, 0.22, 0.01]


def make_sample(n_rows: int = 200_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Diurnal volume: quiet night hours, busy daytime, across 744 hourly steps.
    hour_weight = np.array([0.06] * 9 + [1.0] * 15)
    step_weight = np.tile(hour_weight, 31)[:744]
    step_weight = step_weight / step_weight.sum()
    step = np.sort(rng.choice(np.arange(1, 745), size=n_rows, p=step_weight))

    tx_type = rng.choice(TYPES, size=n_rows, p=TYPE_WEIGHTS)
    amount = np.round(rng.lognormal(mean=8.5, sigma=1.4, size=n_rows), 2)

    # Origins are nearly unique; destinations repeat, merchants most of all.
    orig = np.array([f"C{i:09d}" for i in rng.integers(1, n_rows * 20, size=n_rows)])
    merchants = np.array([f"M{i:09d}" for i in rng.integers(1, 3_000, size=3_000)])
    customers = np.array([f"C{i:09d}" for i in rng.integers(1, 40_000, size=15_000)])
    is_payment = tx_type == "PAYMENT"
    dest = np.where(is_payment,
                    rng.choice(merchants, size=n_rows),
                    rng.choice(customers, size=n_rows))

    old_orig = np.round(amount * rng.uniform(0.5, 6.0, size=n_rows), 2)
    sign = np.where(tx_type == "CASH_IN", 1.0, -1.0)
    new_orig = np.round(np.maximum(old_orig + sign * amount, 0.0), 2)
    old_dest = np.round(rng.lognormal(mean=9.0, sigma=1.5, size=n_rows), 2)
    new_dest = np.round(old_dest + amount, 2)

    # Fraud only in TRANSFER / CASH_OUT, with a mild dependence on amount and
    # destination reuse so the synthetic data is at least learnable.
    eligible = np.isin(tx_type, ["TRANSFER", "CASH_OUT"])
    dest_repeat = pd.Series(dest).map(pd.Series(dest).value_counts()).to_numpy()
    logit = (-4.2
             + 1.30 * (np.log10(amount + 1) - 3.5)
             + 0.05 * np.clip(dest_repeat, 0, 60)
             + rng.normal(0, 0.6, size=n_rows))
    probability = 1 / (1 + np.exp(-logit))
    is_fraud = (eligible & (rng.random(n_rows) < probability)).astype(int)

    # Fraud cancellation artifact: account drained, balances zeroed.
    old_orig = np.where(is_fraud == 1, amount, old_orig)
    new_orig = np.where(is_fraud == 1, 0.0, new_orig)
    old_dest = np.where(is_fraud == 1, 0.0, old_dest)
    new_dest = np.where(is_fraud == 1, 0.0, new_dest)

    is_flagged = ((tx_type == "TRANSFER") & (amount > 200_000) & (is_fraud == 1)).astype(int)

    return pd.DataFrame({
        "step": step, "type": tx_type, "amount": amount, "nameOrig": orig,
        "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig, "nameDest": dest,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "isFraud": is_fraud, "isFlaggedFraud": is_flagged,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_paysim.csv")
    args = parser.parse_args()

    frame = make_sample(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} SYNTHETIC rows to {args.out}")
    print(f"Fraud rows: {int(frame['isFraud'].sum()):,} "
          f"({100 * frame['isFraud'].mean():.3f}%)  [invented data, not findings]")


if __name__ == "__main__":
    main()
