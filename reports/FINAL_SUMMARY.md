# Final project summary

_Generated 2026-09-22 15:07 from measured pipeline output._


## Dataset characteristics

| metric | value |
| --- | --- |
| rows | 6,362,620 |
| columns (raw) | 11 |
| fraud transactions | 8,213 |
| fraud rate | 0.1291% |
| legit per fraud | 774 |
| time span | 743 hourly steps = 31.0 days |
| unique origin accounts | 6,353,307 |
| unique destination accounts | 2,721,804 |

## Data limitations that shaped the design

- Hourly resolution only, so no minute-level velocity exists.
- No geography, device or IP, so no location or impossible-travel features.
- ~30 days total, so no true 30-day rolling baseline for early transactions; an expanding past-only baseline is used instead.
- Origin accounts largely single-use: prior origin history covers 0.15% of rows versus 57.22% for destinations.
- Synthetic data, so no claim about real-world performance is supportable.


## Leakage findings

| column | leakage_risk | primary_model_decision |
| --- | --- | --- |
| oldbalanceOrg | MEDIUM | EXCLUDE |
| newbalanceOrig | HIGH | EXCLUDE |
| oldbalanceDest | MEDIUM | EXCLUDE |
| newbalanceDest | HIGH | EXCLUDE |
| isFlaggedFraud | HIGH | EXCLUDE (kept as a benchmark baseline) |
| amount | LOW | INCLUDE |
| type | LOW | INCLUDE |
| step | LOW (use with care) | INCLUDE as hour_of_day only |

## Behavioural features supported by the dataset

38 features were built, all past-only. Families: amount shape, hour-of-day, origin expanding history, destination expanding history, destination fan-in and unique-sender counts, sender-destination relationship novelty, time-window velocity over `step` (24h, 72h, 168h), and transaction type.


## Models

| model | roc_auc | pr_auc |
| --- | --- | --- |
| logistic_regression | 0.952303 | 0.344862 |
| random_forest | 0.970242 | 0.415589 |

Selected model: **logistic_regression**, chosen by PR-AUC on the validation period, then evaluated once on the untouched test period.


## Financial threshold methodology

1. Score the validation period with the selected model.
2. Sweep thresholds 0.01 to 0.99, recording the full confusion matrix at each.
3. Apply the simulated cost function: FN x $500 + FP x $25.
4. Select the minimum-cost threshold on validation, then report its performance on the test period. Choosing the threshold on the same data used to report it would be a subtler form of the leakage this project is about.


## Minimum-cost threshold under the baseline assumptions

| metric | value |
| --- | --- |
| threshold | 0.06 |
| fraud captured | 26.96% |
| precision | 61.03% |
| false positive rate | 0.0569% |
| legitimate blocked | 733 |
| fraud missed | 3,110 |
| simulated total cost | $1,573,325 |
| simulated cost at t=0.50 | $1,950,675 |
| simulated difference | $377,350 |

**SIMULATED / ESTIMATED UNDER ASSUMPTIONS** Minimum simulated cost under the specified assumptions and evaluation sample only. Not a universally optimal threshold, and not a realised saving.


## Cost sensitivity findings

Across 25 cost scenarios the minimum-cost threshold ranges from 0.00 to 0.08. As missed fraud becomes more expensive relative to customer friction, the threshold falls, because the model should intervene more readily when being wrong in the other direction costs more. The operating point is therefore a function of business economics, not a fixed property of the model.


## Dashboard design

Three Power BI pages fed by `outputs/dashboard_data/`:

1. **Executive risk overview** - KPI cards, fraud trend, fraud by type, threshold-versus-cost curve, capture-versus-friction trade-off.
2. **Fraud operations** - action queue sorted by fraud probability, with risk band, velocity and counterparty context for each alert.
3. **Customer friction** - false positives by type, amount band and time, with the simulated friction cost attached.


## Main business insights

1. Accuracy is unusable as a fraud metric at this base rate.
2. Balance fields are excluded from the primary model because of PaySim's simulator-specific balance and reversal mechanics and the need to keep the authorization-time feature set conservative.3. Sender history barely exists in this dataset, so counterparty behaviour carries the behavioural signal.
4. The intervention threshold is a pricing decision, not a model property.
5. Customer friction is concentrated, so it can be targeted with a softer control instead of accepted as the cost of doing business.


## Resume-ready description

> Built an end-to-end fraud analytics system on 6,362,620 mobile-money transactions: ran a leakage audit that excluded four outcome-contaminated balance fields, engineered 38 past-only behavioural features in SQL and pandas (destination velocity over RANGE windows, counterparty fan-in, sender-destination novelty), trained time-split logistic regression and random forest models reaching PR-AUC 0.344862 on a held-out future period, and replaced the default 0.50 decision threshold with a cost-minimising operating point derived from an explicit false-negative and false-positive cost model, validated across 25 cost scenarios and delivered as a three-page Power BI decision dashboard.


## Interview explanation

**What is the project?** A fintech fraud system where the deliverable is not the classifier but the intervention policy. The model produces a probability; the business decision is where to act on it, and that depends on what each kind of mistake costs.

**Hardest technical decision?** Excluding the balance columns. They give a near-perfect model, and the dataset documentation explains why: fraudulent transactions are cancelled, so those fields describe the aftermath, not the transaction. I quantified the separation, excluded them, and kept a labelled leakage demo to show the size of the illusion.

**Why time-aware splitting?** Every behavioural feature is cumulative. A random split lets the model see later transactions from the same destination account during training and earlier ones at test time, which inflates results and tells you nothing about tomorrow.

**Why is PR-AUC the headline metric?** At a fraud rate below one percent, ROC-AUC's denominator is the enormous legitimate population, so it stays flattering while precision is poor. PR-AUC's no-skill line is the fraud rate itself, which makes improvement legible.

**What would you do with more data?** Device fingerprinting, IP intelligence and true timestamps, in that order. The ceiling here is the observation space, not the algorithm.


## Limitations

See the limitations section of `findings.md`. In short: synthetic data, no geography or device signal, hourly resolution, simulated costs, uncalibrated tree probabilities, and a single held-out period rather than rolling revalidation.


## Future improvements

Real-time scoring, streaming feature computation, device fingerprinting, IP intelligence, merchant risk scoring, graph-based detection over the sender-receiver network, adaptive thresholds driven by live cost telemetry, chargeback-derived cost inputs, investigation feedback loops, model monitoring with drift detection, human-in-the-loop review, and formal model governance. None of these is implemented here.

