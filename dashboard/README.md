# Power BI dashboard specification

Power BI presents decisions that were already made and validated in Python. It
does not recompute analytics. Every table it imports is a flat CSV with a
stable schema, produced by `src/dashboard_export.py`.

## Data sources

Run `python run_pipeline.py` first. It writes these into
`outputs/dashboard_data/`:

| File | Grain | Used by |
| --- | --- | --- |
| `scored_transactions.csv` | one row per test-period transaction | Pages 1, 2, 3 |
| `kpi_summary.csv` | single row | Page 1 cards |
| `threshold_cost_curve.csv` | one row per threshold | Page 1 |
| `cost_sensitivity.csv` | one row per cost scenario (25) | Page 1 detail |
| `fraud_by_type.csv` | one row per transaction type | Pages 1, 3 |
| `fraud_over_time.csv` | one row per `step` | Pages 1, 3 |
| `false_positive_profile.csv` | type x amount band | Page 3 |
| `risk_band_definition.csv` | one row per band (3) | Page 2 legend |

**Import steps:** Get Data → Text/CSV → point at each file → Load. In Power
Query set `fraud_probability` to Decimal Number, `step` and the count columns
to Whole Number, and `risk_band` / `outcome_class` / `type` to Text.

No relationships are needed if you build each visual from its own table.
If you prefer a model, join everything to `scored_transactions` on `type` and
`step`; keep `threshold_cost_curve` and `cost_sensitivity` as disconnected
tables since they are curves, not facts about transactions.

Every monetary column is **SIMULATED under stated assumptions**. Put that
sentence in a text box on Page 1. It is the difference between a portfolio
project and a misleading one.

---

## Page 1 - Executive risk overview

*Question: what is the current fraud-risk picture and what does it cost?*

**KPI cards** (all from `kpi_summary.csv`, single-row table, so use `Max` or
`First` as the aggregation):

- Total transactions
- Total fraud transactions
- Fraud rate %
- Total transaction value
- Fraudulent transaction value
- Fraud capture rate %
- False positive rate %
- Simulated total cost
- Selected threshold

**Visuals**

1. *Fraud trend* - line chart, `fraud_over_time.csv`, axis `step`, values
   `fraud` and `alerts`. Add `transactions` on a secondary axis to show volume
   context.
2. *Fraud by transaction type* - clustered bar, `fraud_by_type.csv`, axis
   `type`, values `fraud` and `false_positives`. Putting caught fraud and
   created friction on the same chart is the whole story of the project in one
   visual.
3. *Transaction amount distribution* - column chart, `scored_transactions.csv`,
   axis `amount_band`, value count of `transaction_id`, legend `isFraud`.
4. *Threshold versus total cost* - line chart, `threshold_cost_curve.csv`,
   axis `threshold`, values `total_cost`, `cost_missed_fraud`,
   `cost_customer_friction`. Filter `grid_source = "fixed"` for an evenly
   spaced axis. Add a constant line at the selected threshold.
5. *Capture versus friction* - line chart, same table, axis `threshold`,
   values `fraud_capture_rate` and `legitimate_transactions_blocked` on a
   secondary axis.

**Useful measures**

```DAX
Fraud Rate % =
DIVIDE( SUM(scored_transactions[isFraud]), COUNTROWS(scored_transactions) )

Alert Precision % =
DIVIDE(
    CALCULATE( COUNTROWS(scored_transactions),
               scored_transactions[outcome_class] = "true_positive" ),
    CALCULATE( COUNTROWS(scored_transactions),
               scored_transactions[predicted_flag] = 1 )
)

Simulated Total Cost =
  CALCULATE( COUNTROWS(scored_transactions),
             scored_transactions[outcome_class] = "false_negative" ) * 500
+ CALCULATE( COUNTROWS(scored_transactions),
             scored_transactions[outcome_class] = "false_positive" ) * 25
```

Replace 500 and 25 with What-If parameters if you want the page interactive:
Modeling → New parameter → Numeric range. Then the cost cards recompute live
and the page becomes a negotiation tool rather than a static report.

---

## Page 2 - Fraud operations

*Question: what should an analyst work on right now?*

**Action queue** - table visual on `scored_transactions.csv`, sorted by
`fraud_probability` descending, filtered to `risk_band` in {HIGH, MEDIUM}.

Columns: `transaction_id`, `step`, `type`, `amount`, `fraud_probability`,
`risk_band`, `dest_txn_prev_24h`, `dest_prior_unique_senders`,
`dest_prior_txn_count`, `pair_is_new`, `dest_is_new`,
`orig_amount_to_prior_avg`, `amount_to_type_prior_avg`.

Conditional formatting: data bars on `fraud_probability`, background colour on
`risk_band`.

**Supporting visuals**

- Card: alerts in queue (count where `predicted_flag = 1`)
- Donut: queue composition by `risk_band`
- Bar: alerts by `type`
- Slicers: `risk_band`, `type`, `amount_band`, `step`

**Risk band legend** - table visual on `risk_band_definition.csv` showing
`band`, `range`, `action`, `rationale`, `fraud_rate_in_band_pct`.

The bands are derived from the cost model, not chosen by eye. A step-up
verification costs less than a hard decline, so it pays for itself at a lower
level of suspicion; running the cost minimisation at both intervention costs
produces two thresholds, and MEDIUM is the region between them. Keep the
`rationale` column visible so a reviewer can see that.

---

## Page 3 - Customer friction

*Question: where are fraud controls creating unnecessary customer friction?*

**Cards**

- Legitimate transactions blocked (`outcome_class = "false_positive"`)
- False positive rate %
- Simulated friction cost
- Value of legitimate transactions blocked

**Visuals**

1. *False positives by transaction type* - bar, `false_positive_profile.csv`,
   axis `type`, value `false_positives`, with `false_positive_rate_pct` as a
   line. Count and rate answer different questions, so show both.
2. *False positives by amount band* - bar, same table, axis `amount_band`.
3. *False positives over time* - line, `fraud_over_time.csv`, axis `step`,
   value `false_positive`.
4. *Friction heat* - matrix, rows `type`, columns `amount_band`, values
   `false_positive_rate_pct`, with conditional formatting. This is the visual
   that tells an operations lead exactly which segment to move from hard
   decline to step-up verification.
5. *Blocked value at risk* - card or bar of `SUM(amount)` filtered to
   `outcome_class = "false_positive"`. A blocked $200,000 transfer is not the
   same customer event as a blocked $20 payment, and the cost model treats them
   identically. Showing the value makes that limitation visible rather than
   hidden.

---

## Building the .pbix

This repository ships the data and the specification, not a binary .pbix, so
the build stays reproducible and reviewable in git. Build the file locally and
export a PDF or screenshots into `dashboard/` for the README if you want the
visuals visible on GitHub.

Two things worth doing before you screenshot it:

1. Put the SIMULATED-assumptions note where a reader cannot miss it.
2. Label the selected threshold on the cost chart with how it was chosen
   (minimum simulated cost on the validation period, reported on test).
