-- ===========================================================================
-- 04_analysis_queries.sql  |  Business analysis on the feature layer
--
-- Run:  psql -d fraud_analytics -f sql/04_analysis_queries.sql
--
-- Each query answers a question a fraud lead would actually ask. These are
-- descriptive: they show where fraud concentrates and how behavioural features
-- differ between classes. None of them is a model, and none of them proves
-- causation.
-- ===========================================================================

\echo '=== Q1. Does each behavioural feature separate fraud from legitimate? ==='

-- If a feature's fraud and legitimate medians are nearly identical, it will
-- carry little weight in any model. This is a cheap sanity check before
-- spending an hour training.
SELECT
    CASE WHEN is_fraud = 1 THEN 'fraudulent' ELSE 'legitimate' END AS class,
    COUNT(*)                                                        AS transactions,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)::NUMERIC, 2)
                                                                    AS median_amount,
    ROUND(AVG(dest_prior_txn_count), 2)                             AS avg_dest_prior_txns,
    ROUND(AVG(dest_prior_unique_senders), 2)                        AS avg_dest_fan_in,
    ROUND(AVG(dest_txn_prev_24h), 3)                                AS avg_dest_velocity_24h,
    ROUND(AVG(dest_txn_prev_168h), 3)                               AS avg_dest_velocity_168h,
    ROUND(AVG(pair_prior_txn_count), 3)                             AS avg_pair_history,
    ROUND(100.0 * AVG(pair_is_new), 2)                              AS pct_new_relationship,
    ROUND(100.0 * AVG(dest_is_new), 2)                              AS pct_new_destination,
    ROUND(100.0 * AVG(dest_is_merchant), 2)                         AS pct_to_merchant,
    ROUND(AVG(amount_to_type_prior_avg)::NUMERIC, 3)                AS avg_amount_vs_type_avg
FROM transaction_features
GROUP BY 1;


\echo '=== Q2. Where does fraud concentrate by type and hour? ==='

SELECT
    txn_type,
    COUNT(*)                                        AS transactions,
    SUM(is_fraud)                                   AS fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
    ROUND(SUM(amount) FILTER (WHERE is_fraud = 1), 2) AS fraud_value,
    ROUND(100.0 * SUM(amount) FILTER (WHERE is_fraud = 1)
          / NULLIF(SUM(amount), 0), 4)              AS fraud_share_of_value_pct
FROM transaction_features
GROUP BY txn_type
ORDER BY fraud DESC;

-- Hour-of-day, restricted to the types that actually carry fraud, so the
-- pattern is not diluted by categories with a zero base rate.
SELECT
    hour_of_day,
    COUNT(*)                                   AS transactions,
    SUM(is_fraud)                              AS fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM transaction_features
WHERE txn_type IN ('TRANSFER', 'CASH_OUT')
GROUP BY hour_of_day
ORDER BY hour_of_day;


\echo '=== Q3. Amount bands: where is the risk per transaction highest? ==='

WITH banded AS (
    SELECT
        CASE
            WHEN amount <   1000 THEN '1. under 1k'
            WHEN amount <  10000 THEN '2. 1k-10k'
            WHEN amount <  50000 THEN '3. 10k-50k'
            WHEN amount < 200000 THEN '4. 50k-200k'
            WHEN amount < 1000000 THEN '5. 200k-1M'
            ELSE                      '6. 1M+'
        END AS amount_band,
        is_fraud,
        amount
    FROM transaction_features
)
SELECT
    amount_band,
    COUNT(*)                                   AS transactions,
    SUM(is_fraud)                              AS fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount) FILTER (WHERE is_fraud = 1), 2) AS fraud_value,
    -- Value at risk per 1,000 transactions reviewed in this band. This is the
    -- band-level version of the review-capacity argument.
    ROUND(1000.0 * SUM(amount) FILTER (WHERE is_fraud = 1) / COUNT(*), 2)
                                               AS fraud_value_per_1000_txns
FROM banded
GROUP BY amount_band
ORDER BY amount_band;


\echo '=== Q4. Destination fan-in: are heavily-fanned-in receivers riskier? ==='

WITH banded AS (
    SELECT
        CASE
            WHEN dest_prior_unique_senders = 0  THEN '0 senders (first ever)'
            WHEN dest_prior_unique_senders < 5  THEN '1-4 senders'
            WHEN dest_prior_unique_senders < 20 THEN '5-19 senders'
            WHEN dest_prior_unique_senders < 50 THEN '20-49 senders'
            ELSE                                     '50+ senders'
        END AS fan_in_band,
        MIN(dest_prior_unique_senders) AS band_floor,
        COUNT(*)                                   AS transactions,
        SUM(is_fraud)                              AS fraud
    FROM transaction_features
    GROUP BY 1
)
SELECT fan_in_band, transactions, fraud,
       ROUND(100.0 * fraud / transactions, 4) AS fraud_rate_pct
FROM banded
ORDER BY band_floor;


\echo '=== Q5. Relationship novelty: first payment to a counterparty ==='

SELECT
    CASE WHEN pair_is_new = 1 THEN 'first transaction to this counterparty'
         ELSE 'existing relationship' END       AS relationship,
    COUNT(*)                                    AS transactions,
    SUM(is_fraud)                               AS fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)  AS fraud_rate_pct
FROM transaction_features
GROUP BY 1;


\echo '=== Q6. Velocity: does a burst of activity to one receiver precede fraud? ==='

WITH banded AS (
    SELECT
        CASE
            WHEN dest_txn_prev_24h = 0  THEN '0'
            WHEN dest_txn_prev_24h = 1  THEN '1'
            WHEN dest_txn_prev_24h < 5  THEN '2-4'
            WHEN dest_txn_prev_24h < 10 THEN '5-9'
            ELSE                             '10+'
        END AS velocity_band,
        MIN(dest_txn_prev_24h) AS band_floor,
        COUNT(*)               AS transactions,
        SUM(is_fraud)          AS fraud
    FROM transaction_features
    GROUP BY 1
)
SELECT velocity_band AS dest_txns_in_prev_24h, transactions, fraud,
       ROUND(100.0 * fraud / transactions, 4) AS fraud_rate_pct
FROM banded
ORDER BY band_floor;


\echo '=== Q7. Daily trend: is fraud stable across the simulated month? ==='

SELECT
    sim_day,
    COUNT(*)                                   AS transactions,
    SUM(is_fraud)                              AS fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount) FILTER (WHERE is_fraud = 1), 2) AS fraud_value
FROM transaction_features
GROUP BY sim_day
ORDER BY sim_day;


\echo '=== Q8. Time-aware split boundaries (must match src/train_models.py) ==='

-- The Python pipeline splits chronologically at 60% / 80% of transaction
-- volume. This query reproduces the boundaries so the SQL layer and the model
-- layer can be reconciled.
WITH cumulative AS (
    SELECT
        step,
        COUNT(*) AS txns,
        SUM(COUNT(*)) OVER (ORDER BY step)::NUMERIC
            / SUM(COUNT(*)) OVER () AS cumulative_share
    FROM transaction_features
    GROUP BY step
)
SELECT
    MAX(step) FILTER (WHERE cumulative_share <= 0.60) AS train_end_step,
    MAX(step) FILTER (WHERE cumulative_share <= 0.80) AS validation_end_step,
    MAX(step)                                         AS test_end_step
FROM cumulative;


\echo '=== Q9. Top-risk review queue by rule (pre-model reference) ==='

-- What an analyst queue looks like WITHOUT a model: the largest transfers to
-- brand-new counterparties. Useful as the comparison point for the Stage 3
-- model-driven queue in outputs/dashboard_data/scored_transactions.csv.
SELECT
    txn_id, step, txn_type, amount,
    dest_prior_txn_count,
    dest_prior_unique_senders,
    dest_txn_prev_24h,
    pair_is_new,
    is_fraud
FROM transaction_features
WHERE txn_type IN ('TRANSFER', 'CASH_OUT')
  AND pair_is_new = 1
ORDER BY amount DESC
LIMIT 50;
