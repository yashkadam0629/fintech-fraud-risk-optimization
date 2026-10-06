# Findings

_Generated 2026-09-22 15:07 by `run_pipeline.py`. Every number below is read from measured pipeline output._


_Data source: `C:\Users\Yash Kadam\Downloads\fintechfinal\project\fintech-fraud-optimization\data\raw\PS_20174392719_1491204439457_log.csv`_

_Rows analysed: 6,362,620_


---

## 1. Fraud is rare, so accuracy is the wrong yardstick

**OBSERVATION.** 8,213 of 6,362,620 transactions are labelled fraudulent, a rate of 0.1291%, roughly one in 774. A classifier that predicts 'legitimate' every single time scores 99.871% accuracy.

**INTERPRETATION.** Accuracy is dominated by the majority class and carries almost no information about fraud performance. PR-AUC, whose no-skill baseline equals the fraud rate, is the appropriate headline ranking metric, with precision and recall reported at a specific operating point.

**BUSINESS IMPLICATION.** Any vendor or internal dashboard reporting fraud-model 'accuracy' is reporting a number that cannot distinguish a working control from no control at all.

**RECOMMENDATION.** Report fraud capture rate, false-positive rate and alert volume at a named threshold. Retire accuracy from fraud reporting.


## 2. Balance columns leak the target

**OBSERVATION.** Measured separation between fraudulent and legitimate transactions on balance-pattern checks:

| check | fraud % | legitimate % | gap (pp) |
| --- | --- | --- | --- |
| origin balance does not reconcile | 0.547912 | 59.48 | -58.93 |
| destination balance does not reconcile | 56.50 | 65.84 | -9.35 |
| amount equals entire origin balance | 97.82 | 1.57371e-05 | 97.82 |
| origin balance emptied to zero | 98.05 | 56.68 | 41.37 |
| destination balances both zero | 49.56 | 36.40 | 13.15 |

**INTERPRETATION.** PaySim provides pre-transaction balances (`oldbalanceOrg` and `oldbalanceDest`) and post-transaction balances (`newbalanceOrig` and `newbalanceDest`). The primary model excludes all balance fields as a conservative design choice because the simulator's balance and reversal mechanics can create signals that are difficult to interpret as clean authorization-time behavioural evidence. In particular, the `newbalance*` fields are post-transaction values and would not be available when an authorization decision is made.

**BUSINESS IMPLICATION.** The balance fields are therefore kept outside the primary model. The `oldbalance*` fields are pre-transaction and could be evaluated in a separate model, while the `newbalance*` fields are unsuitable for authorization-time scoring because they describe the transaction after it occurs.

**RECOMMENDATION.** Exclude all four balance columns from the primary model as a conservative feature-design choice, and exclude `isFlaggedFraud` because it is a simulator control output rather than an independent behavioural observation. Keep the leakage demonstration clearly labelled and separate from the primary evaluation.


### Leakage decision table

| column | leakage_risk | reason | primary_model_decision |
| --- | --- | --- | --- |
| oldbalanceOrg | MEDIUM | Pre-transaction origin balance. It is available before the transaction, but is excluded from the primary model conservatively because PaySim's balance and reversal mechanics can create simulator-specific behavioural signals. | EXCLUDE |
| newbalanceOrig | HIGH | Post-transaction origin balance. It describes the account after the transaction and is therefore not available when the authorization decision is made. | EXCLUDE |
| oldbalanceDest | MEDIUM | Pre-transaction destination balance. It is available before the transaction, but is excluded from the primary model conservatively because of PaySim's simulator-specific balance and reversal mechanics. | EXCLUDE |
| newbalanceDest | HIGH | Post-transaction destination balance. It describes the destination account after the transaction and is therefore not available at authorization time. | EXCLUDE |
| isFlaggedFraud | HIGH | A control output, not an independent behavioural observation. It is excluded from the primary model because it represents a simulator-generated fraud flag rather than a clean input available for independent modelling. | EXCLUDE (kept as a benchmark baseline) |
| amount | LOW | Known before the decision is made; a genuine input. | INCLUDE |
| type | LOW | Known at request time. | INCLUDE |
| step | LOW (use with care) | Known at request time, but the raw index encodes position in the simulation and does not generalise forward. Only hour-of-day is used as a feature. | INCLUDE as hour_of_day only |

## 3. Originating accounts barely repeat, which reshapes the feature design

**OBSERVATION.** There are 6,353,307 unique originating accounts across the dataset. Only 0.15% of them appear more than once, accounting for 0.29% of transactions. Prior origin history exists for 0.15% of rows, against 57.22% for destination accounts.

**INTERPRETATION.** Per-customer behavioural baselines such as 'amount versus this customer's historical average' are undefined for the large majority of transactions on the sender side. Destination accounts, by contrast, accumulate usable history.

**BUSINESS IMPLICATION.** A feature set built around sender history would be mostly missing values dressed up as signal. Counterparty behaviour is where the observable structure actually lives in this dataset.

**RECOMMENDATION.** Base behavioural features on destination history, sender-destination relationship novelty, destination fan-in and time-window velocity. Compute origin features too, but publish their coverage alongside them rather than presenting them as fully populated.


## 4. Baselines the model has to beat

| baseline | TP | FP | FN | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 flag nothing | 0 | 0 | 8,213 | 0 | 0 | 0 |
| 2 flag everything | 8,213 | 6,354,407 | 0 | 0.00129082 | 1.00 | 0.00257831 |
| 3 isFlaggedFraud | 16 | 0 | 8,197 | 1.00 | 0.00194813 | 0.00388869 |
| 4 transfer or cashout | 8,213 | 2,762,196 | 0 | 0.00296454 | 1.00 | 0.00591156 |
| 5 amount above p99 | 1,969 | 61,658 | 6,244 | 0.030946 | 0.239742 | 0.0548163 |

Amount used as a continuous score: ROC-AUC 0.789921, PR-AUC 0.0186966 against a no-skill baseline of 0.00129082.


## 5. Model performance on a held-out future period

**OBSERVATION.** Models were trained on the earliest 59.61% of transactions by `step`, tuned on the next 20.06%, and evaluated on the final 20.33%, which the models never saw. On that test period the selected model (`logistic_regression`) reached PR-AUC 0.344862 and ROC-AUC 0.952303.

**INTERPRETATION.** A chronological split reproduces the only question that matters operationally: does a model fitted on the past work on traffic it has not seen. A random split would have leaked future behaviour into cumulative features and inflated these numbers.

**BUSINESS IMPLICATION.** The ranking quality is what determines whether a review queue can be prioritised at all. The threshold decision in the next section is only meaningful because the ranking carries signal.

**RECOMMENDATION.** Re-validate on a rolling forward window before any deployment, and monitor PR-AUC drift by month rather than a single static score.


## 6. The cost-minimising intervention threshold

**OBSERVATION.** Sweeping thresholds from 0.01 to 0.99 under the simulated assumptions (missed fraud $500, blocked legitimate transaction $25), total simulated cost is minimised at threshold **0.06**, giving 1,148 true positives, 733 false positives, 3,110 missed fraud cases, fraud capture 26.96% and a false-positive rate of 0.0569%. Simulated total cost $1,573,325 versus $1,950,675 at the conventional 0.50 threshold.

**INTERPRETATION.** 0.50 is a mathematical convention that implicitly assumes a missed fraud and an inconvenienced customer cost the same. Once the two costs are stated explicitly, the cost curve has a different minimum.

**BUSINESS IMPLICATION.** Under these assumptions the threshold choice alone moves simulated cost by $377,350 (19.34%) on this evaluation sample, with no change to the model itself.

**RECOMMENDATION.** Treat the threshold as a governed business parameter with a named owner and a review cadence, not a default buried in code. **SIMULATED / ESTIMATED UNDER ASSUMPTIONS** These are not measured company savings.


## 7. The threshold depends on economics, not just the model

**OBSERVATION.** Across 25 cost scenarios the minimum-cost threshold ranges from 0.00223128 to 0.08 (median 0.0138814). Fraud capture across those scenarios ranges from 22.12% to 82.34%.

**INTERPRETATION.** Across 25 cost scenarios the minimum-cost threshold ranges from 0.00 to 0.08. As missed fraud becomes more expensive relative to customer friction, the threshold falls, because the model should intervene more readily when being wrong in the other direction costs more. The operating point is therefore a function of business economics, not a fixed property of the model.

**BUSINESS IMPLICATION.** Two teams running the identical model can be correct to operate at very different thresholds if their cost structures differ. Disagreement about aggressiveness is usually disagreement about costs, not about the model.

**RECOMMENDATION.** Agree the cost of a blocked legitimate transaction with Finance and Customer Operations first. Re-run this grid whenever chargeback costs or support economics change.


## 8. What the model is responding to

| feature | importance | importance_type |
| --- | --- | --- |
| type_CASH_OUT | 1.14 | absolute_standardised_coefficient |
| type_TRANSFER | 1.09 | absolute_standardised_coefficient |
| type_PAYMENT | 0.955777 | absolute_standardised_coefficient |
| dest_is_merchant | 0.955603 | absolute_standardised_coefficient |
| type_CASH_IN | 0.860524 | absolute_standardised_coefficient |
| hour_of_day | 0.632452 | absolute_standardised_coefficient |
| dest_is_new | 0.516789 | absolute_standardised_coefficient |
| dest_has_history | 0.516788 | absolute_standardised_coefficient |
| type_DEBIT | 0.499043 | absolute_standardised_coefficient |
| amount_to_type_prior_avg | 0.368354 | absolute_standardised_coefficient |

These features contributed to the model's scores. They are associated with the prediction; none of this establishes causation.


## 9. Limitations

- PaySim is synthetic. Patterns come from a simulator, so performance here does not transfer to a real payments portfolio.
- The dataset has no sub-hourly timing, no geography, no device or IP data, and no merchant category, so whole families of production fraud features cannot be evaluated at all.
- All monetary figures are simulated. The dataset contains no recovery rates, chargeback costs or investigation costs.
- The cost model treats every false positive as equally costly. In reality, blocking a large recurring payment for a long-tenured customer is not equivalent to declining a small one-off.
- Tree-model probabilities are not calibrated, so threshold values are not directly comparable between models even when both are well ranked.
- A single held-out period is one sample. Confidence intervals and rolling revalidation are absent.

