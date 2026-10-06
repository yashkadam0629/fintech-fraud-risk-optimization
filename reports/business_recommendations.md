# Business recommendations

_Generated 2026-09-22 15:07. All monetary figures are **SIMULATED / ESTIMATED UNDER ASSUMPTIONS** and are not measured company results._


## 1. Operate a three-tier intervention policy, not a single block rule

The cost model produces two different thresholds because two different interventions have two different costs. A step-up verification annoys a customer less than a hard decline, so it pays for itself at a lower level of suspicion. That gives a derived band structure rather than round numbers chosen by eye:

| band | range | action | rationale |
| --- | --- | --- | --- |
| LOW | score < 0.01 | allow without friction | Below this score, the expected fraud loss prevented is smaller than even the cheap verification cost. |
| MEDIUM | 0.01 <= score < 0.06 | step-up verification / manual review | Intervening pays off at a friction cost of $5 but not at $25. |
| HIGH | score >= 0.06 | block / hard decline | Expected fraud loss exceeds the full blocking cost under the stated assumptions. |

Boundaries: verify at **0.0134852**, block at **0.06**, derived from friction costs of $5 and $25 against a missed-fraud cost of $500.


## 2. Concentrate controls where fraud actually occurs

**OBSERVATION.** Labelled fraud appears only in these transaction types: CASH_OUT, TRANSFER. These types carry no labelled fraud at all: CASH_IN, DEBIT, PAYMENT.

**INTERPRETATION.** In this dataset the account-takeover pattern is expressed through specific movement types. Applying identical scrutiny to every type spends review capacity where the base rate is effectively zero.

**BUSINESS IMPLICATION.** Uniform controls create customer friction in categories with no measured fraud exposure, which is pure cost.

**RECOMMENDATION.** Scope the intervention policy to the fraud-bearing types first. Maintain light monitoring elsewhere to detect a shift in the pattern rather than assuming it is permanent.


## 3. Target the friction you are creating

**OBSERVATION.** At the selected threshold, 733 legitimate transactions are blocked, a false-positive rate of 0.0569%, carrying a simulated friction cost of $18,325. The heaviest concentration is in TRANSFER (719 cases, 0.6550%).

**INTERPRETATION.** False positives are not spread evenly. They cluster where the model is least discriminative, which is usually where legitimate behaviour most resembles the fraud pattern.

**BUSINESS IMPLICATION.** Blanket blocking in that segment buys relatively little fraud prevention per disrupted customer.

**RECOMMENDATION.** Route that segment to step-up verification rather than a hard decline, and track the recovery rate of verified transactions as the measure of whether the softer control is working.


## 4. Govern the threshold as a business parameter

**OBSERVATION.** The minimum-cost threshold moved across the sensitivity grid from 0.00223128 to 0.08 purely as a function of the cost assumptions.

**INTERPRETATION.** The threshold is not a model property. It is a pricing decision about the relative cost of two kinds of error.

**BUSINESS IMPLICATION.** If nobody owns the cost assumptions, nobody owns the threshold, and it drifts to whatever the default in the code happens to be.

**RECOMMENDATION.** Assign ownership of the two cost inputs to Finance and Customer Operations, review them quarterly, and re-run the sensitivity grid on each revision.


## 5. Invest in the data that is missing before tuning the model further

**OBSERVATION.** The dataset offers no device, IP, geography, merchant-category or sub-hourly timing information, and originating accounts are largely single-use.

**INTERPRETATION.** The remaining headroom in this system is limited less by algorithm choice than by the narrowness of the observation space.

**BUSINESS IMPLICATION.** Further model tuning on the same fields has a low ceiling; new signal sources have a much higher one.

**RECOMMENDATION.** Prioritise device fingerprinting, IP intelligence, precise timestamps and counterparty risk scoring in the data roadmap ahead of further model experimentation.


---

_All cost figures in this document are **SIMULATED / ESTIMATED UNDER ASSUMPTIONS**. They describe modelled outcomes on a synthetic dataset under stated assumptions and are not realised savings for any company._

