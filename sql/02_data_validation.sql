-- ===========================================================================
-- 02_data_validation.sql  |  Audit the loaded data before trusting it
--
-- Run:  psql -d fraud_analytics -f sql/02_data_validation.sql
--
-- These queries mirror src/data_profiling.py. Running both and comparing is a
-- genuine check: if the SQL and the Python disagree, one of them is wrong, and
-- finding that out now is much cheaper than finding out from a dashboard.
-- ===========================================================================

\echo '=== 1. Shape, nulls and duplicates ==='

SELECT
    COUNT(*)                                             AS total_rows,
    COUNT(*) FILTER (WHERE amount IS NULL)               AS null_amount,
    COUNT(*) FILTER (WHERE old_balance_orig IS NULL)     AS null_old_balance_orig,
    COUNT(*) FILTER (WHERE new_balance_dest IS NULL)     AS null_new_balance_dest,
    COUNT(*) FILTER (WHERE amount = 0)                   AS zero_amount,
    COUNT(*) FILTER (WHERE amount < 0)                   AS negative_amount
FROM raw_transactions;

-- Exact duplicates across every business column. These may be genuine repeat
-- payments inside one simulated hour, so they are reported, not deleted.
WITH duplicate_groups AS (
    SELECT step, txn_type, amount, name_orig, name_dest,
           old_balance_orig, new_balance_orig,
           old_balance_dest, new_balance_dest, is_fraud,
           COUNT(*) AS occurrences
    FROM raw_transactions
    GROUP BY 1,2,3,4,5,6,7,8,9,10
    HAVING COUNT(*) > 1
)
SELECT
    COUNT(*)                          AS duplicate_groups,
    COALESCE(SUM(occurrences - 1), 0) AS surplus_rows
FROM duplicate_groups;


\echo '=== 2. Class balance, overall and by transaction type ==='

SELECT
    COUNT(*)                                   AS transactions,
    SUM(is_fraud)                              AS fraud,
    COUNT(*) - SUM(is_fraud)                   AS legitimate,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND((COUNT(*) - SUM(is_fraud))::NUMERIC
          / NULLIF(SUM(is_fraud), 0), 1)       AS legit_per_fraud,
    -- The accuracy a model gets for free by never predicting fraud.
    ROUND(100.0 * (COUNT(*) - SUM(is_fraud)) / COUNT(*), 4) AS always_legit_accuracy_pct
FROM raw_transactions;

SELECT
    txn_type,
    COUNT(*)                                   AS transactions,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_volume,
    SUM(is_fraud)                              AS fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(100.0 * SUM(is_fraud) / NULLIF(SUM(SUM(is_fraud)) OVER (), 0), 2)
                                               AS share_of_all_fraud_pct,
    ROUND(AVG(amount), 2)                      AS mean_amount,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)::NUMERIC, 2) AS median_amount
FROM raw_transactions
GROUP BY txn_type
ORDER BY fraud DESC;


\echo '=== 3. Amount distribution, fraud vs legitimate ==='

SELECT
    CASE WHEN is_fraud = 1 THEN 'fraudulent' ELSE 'legitimate' END AS class,
    COUNT(*)                                                        AS transactions,
    ROUND(AVG(amount), 2)                                           AS mean_amount,
    ROUND(STDDEV_SAMP(amount), 2)                                   AS std_amount,
    ROUND(MIN(amount), 2)                                           AS min_amount,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount)::NUMERIC, 2) AS q1,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount)::NUMERIC, 2) AS median,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount)::NUMERIC, 2) AS q3,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount)::NUMERIC, 2) AS p99,
    ROUND(MAX(amount), 2)                                           AS max_amount,
    ROUND(SUM(amount), 2)                                           AS total_value
FROM raw_transactions
GROUP BY 1;


\echo '=== 4. Time structure: step, day and hour-of-day ==='

SELECT
    MIN(step)                     AS first_step,
    MAX(step)                     AS last_step,
    COUNT(DISTINCT step)          AS distinct_steps,
    ROUND(COUNT(*)::NUMERIC / COUNT(DISTINCT step), 1) AS mean_txns_per_step,
    ROUND((MAX(step) - MIN(step) + 1) / 24.0, 1)       AS implied_days
FROM raw_transactions;

-- Hour-of-day profile. Watch for the simulator artifact: if legitimate volume
-- collapses overnight while fraud continues at a steady rate, hour-of-day
-- becomes an unrealistically strong predictor.
SELECT
    (step - 1) % 24                            AS hour_of_day,
    COUNT(*)                                   AS transactions,
    SUM(is_fraud)                              AS fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM raw_transactions
GROUP BY 1
ORDER BY 1;


\echo '=== 5. Account repetition: can customer-level history exist at all? ==='

WITH origin_counts AS (
    SELECT name_orig, COUNT(*) AS txns
    FROM raw_transactions GROUP BY name_orig
)
SELECT
    'origin'                                                  AS side,
    COUNT(*)                                                  AS unique_accounts,
    COUNT(*) FILTER (WHERE txns > 1)                          AS repeat_accounts,
    ROUND(100.0 * COUNT(*) FILTER (WHERE txns > 1) / COUNT(*), 2) AS pct_accounts_repeat,
    ROUND(100.0 * SUM(txns) FILTER (WHERE txns > 1) / SUM(txns), 2) AS pct_txns_from_repeat,
    ROUND(AVG(txns), 3)                                       AS mean_txns_per_account,
    MAX(txns)                                                 AS max_txns_one_account
FROM origin_counts
UNION ALL
SELECT
    'destination',
    COUNT(*),
    COUNT(*) FILTER (WHERE txns > 1),
    ROUND(100.0 * COUNT(*) FILTER (WHERE txns > 1) / COUNT(*), 2),
    ROUND(100.0 * SUM(txns) FILTER (WHERE txns > 1) / SUM(txns), 2),
    ROUND(AVG(txns), 3),
    MAX(txns)
FROM (SELECT name_dest, COUNT(*) AS txns FROM raw_transactions GROUP BY name_dest) d;

-- Destination fan-in: how many DISTINCT senders each receiver sees.
SELECT
    ROUND(AVG(unique_senders), 2)  AS mean_unique_senders,
    MAX(unique_senders)            AS max_unique_senders,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY unique_senders)::NUMERIC, 1)
                                   AS p99_unique_senders
FROM (
    SELECT name_dest, COUNT(DISTINCT name_orig) AS unique_senders
    FROM raw_transactions GROUP BY name_dest
) fan_in;

-- Counterparty kind. In PaySim the first character encodes it: C = customer,
-- M = merchant. This is real structure in the data, not an assumption.
SELECT
    LEFT(name_dest, 1)                         AS dest_kind,
    COUNT(*)                                   AS transactions,
    SUM(is_fraud)                              AS fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM raw_transactions
GROUP BY 1;


\echo '=== 6. Leakage audit: do the balance fields encode the outcome? ==='

SELECT
    CASE WHEN is_fraud = 1 THEN 'fraudulent' ELSE 'legitimate' END AS class,
    COUNT(*)                                                        AS transactions,
    -- Origin balance does not reconcile with the movement
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE ABS(new_balance_orig
                  - (old_balance_orig + CASE WHEN txn_type = 'CASH_IN'
                                             THEN amount ELSE -amount END)) > 0.01
    ) / COUNT(*), 4) AS origin_not_reconciled_pct,
    -- Receiver balance does not reconcile
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE ABS(new_balance_dest - (old_balance_dest + amount)) > 0.01
    ) / COUNT(*), 4) AS dest_not_reconciled_pct,
    -- The account was drained exactly: amount equals the entire prior balance
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE ABS(old_balance_orig - amount) < 0.01
    ) / COUNT(*), 4) AS amount_equals_full_balance_pct,
    -- Receiver balances both zero despite a positive amount
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE old_balance_dest = 0 AND new_balance_dest = 0 AND amount > 0
    ) / COUNT(*), 4) AS dest_balances_both_zero_pct
FROM raw_transactions
GROUP BY 1;

-- A wide gap between the two rows above means the column is reporting the
-- consequence of the fraud, not a signal available before the decision.
-- Those columns are excluded from the model. See reports/findings.md.


\echo '=== 7. isFlaggedFraud: the simulator control rule ==='

SELECT
    COUNT(*) FILTER (WHERE is_flagged_fraud = 1)                   AS flagged,
    COUNT(*) FILTER (WHERE is_flagged_fraud = 1 AND is_fraud = 1)  AS flagged_and_fraud,
    COUNT(*) FILTER (WHERE is_flagged_fraud = 1 AND is_fraud = 0)  AS flagged_and_legitimate,
    COUNT(*) FILTER (WHERE is_flagged_fraud = 0 AND is_fraud = 1)  AS fraud_not_flagged,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_flagged_fraud = 1 AND is_fraud = 1)
          / NULLIF(SUM(is_fraud), 0), 4)                           AS pct_of_fraud_flagged
FROM raw_transactions;


\echo '=== 8. Baseline rules the ML model must beat ==='

-- Baseline 4: flag every TRANSFER and CASH_OUT.
-- Baseline 5: flag everything above the 99th percentile of amount.
WITH cutoff AS (
    SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) AS p99 FROM raw_transactions
),
scored AS (
    SELECT
        r.is_fraud,
        CASE WHEN r.txn_type IN ('TRANSFER', 'CASH_OUT') THEN 1 ELSE 0 END AS rule_type,
        CASE WHEN r.amount >= c.p99 THEN 1 ELSE 0 END                      AS rule_amount
    FROM raw_transactions r CROSS JOIN cutoff c
),
confusion AS (
    SELECT 'transfer_or_cashout' AS rule,
           COUNT(*) FILTER (WHERE rule_type = 1 AND is_fraud = 1) AS tp,
           COUNT(*) FILTER (WHERE rule_type = 1 AND is_fraud = 0) AS fp,
           COUNT(*) FILTER (WHERE rule_type = 0 AND is_fraud = 1) AS fn,
           COUNT(*) FILTER (WHERE rule_type = 0 AND is_fraud = 0) AS tn
    FROM scored
    UNION ALL
    SELECT 'amount_above_p99',
           COUNT(*) FILTER (WHERE rule_amount = 1 AND is_fraud = 1),
           COUNT(*) FILTER (WHERE rule_amount = 1 AND is_fraud = 0),
           COUNT(*) FILTER (WHERE rule_amount = 0 AND is_fraud = 1),
           COUNT(*) FILTER (WHERE rule_amount = 0 AND is_fraud = 0)
    FROM scored
)
SELECT
    rule, tp, fp, fn, tn,
    ROUND(tp::NUMERIC / NULLIF(tp + fp, 0), 4) AS precision,
    ROUND(tp::NUMERIC / NULLIF(tp + fn, 0), 4) AS recall,
    ROUND(2.0 * tp / NULLIF(2 * tp + fp + fn, 0), 4) AS f1,
    ROUND(fp::NUMERIC / NULLIF(fp + tn, 0), 6) AS false_positive_rate
FROM confusion;
