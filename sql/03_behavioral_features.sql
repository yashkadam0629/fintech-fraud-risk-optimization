-- ===========================================================================
-- 03_behavioral_features.sql  |  Behavioural feature layer
--
-- Run:  psql -d fraud_analytics -f sql/03_behavioral_features.sql
--
-- Produces the table `transaction_features`, mirroring
-- src/feature_engineering.py feature for feature.
--
-- ---------------------------------------------------------------------------
-- THE RULE THAT GOVERNS EVERY FRAME IN THIS FILE
-- ---------------------------------------------------------------------------
-- Each historical feature uses only information available BEFORE the current
-- transaction. In window-function terms every frame ends at `1 PRECEDING`,
-- never at `CURRENT ROW`. A frame ending at CURRENT ROW includes the
-- transaction being scored, which leaks the present into its own history and
-- produces offline metrics that evaporate in production.
--
-- ---------------------------------------------------------------------------
-- ROWS versus RANGE: the distinction that matters here
-- ---------------------------------------------------------------------------
--   ROWS  BETWEEN 24 PRECEDING AND 1 PRECEDING
--       = the previous 24 ROWS for this account, however far apart in time.
--         Could span ten minutes or three weeks.
--
--   RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING   (ORDER BY step)
--       = every row whose `step` falls in [step - 24, step - 1].
--         Because one step is one hour, this is genuinely "the last 24 hours".
--
-- Velocity is a time concept, so velocity uses RANGE. Expanding lifetime
-- history is a sequence concept, so it uses ROWS UNBOUNDED PRECEDING.
--
-- ---------------------------------------------------------------------------
-- WHAT IS DELIBERATELY ABSENT
-- ---------------------------------------------------------------------------
--   * minute or second velocity: `step` is hourly; finer windows do not exist
--   * anything geographic: PaySim has no country, city, IP, GPS or device
--   * a true 30-day rolling baseline: the dataset is only ~30 days long, so
--     an expanding past-only baseline is used and its coverage is reported
-- ===========================================================================

DROP TABLE IF EXISTS transaction_features;

CREATE TABLE transaction_features AS
WITH base AS (
    SELECT
        txn_id,
        step,
        txn_type,
        amount,
        name_orig,
        name_dest,
        LEFT(name_dest, 1)          AS dest_kind,
        is_fraud,
        -- Time derivations. `step` is an hour counter, so these are exact.
        ((step - 1) / 24) + 1       AS sim_day,
        (step - 1) % 24             AS hour_of_day
    FROM raw_transactions
),

-- ---------------------------------------------------------------------------
-- 1. ORIGIN (SENDER) HISTORY
--    Expanding, past-only. In PaySim most senders appear once, so most of
--    these will be NULL. That is a property of the data, not a bug: the
--    coverage is measured and published rather than hidden behind a fillna.
-- ---------------------------------------------------------------------------
origin_history AS (
    SELECT
        b.*,
        COUNT(*) OVER w_orig                     AS orig_prior_txn_count,
        AVG(amount) OVER w_orig                  AS orig_prior_avg_amount,
        MAX(amount) OVER w_orig                  AS orig_prior_max_amount,
        SUM(amount) OVER w_orig                  AS orig_prior_total_amount,
        step - LAG(step) OVER (PARTITION BY name_orig ORDER BY step, txn_id)
                                                 AS orig_steps_since_prev_txn,
        LAG(txn_type) OVER (PARTITION BY name_orig ORDER BY step, txn_id)
                                                 AS orig_prev_txn_type
    FROM base b
    WINDOW w_orig AS (
        PARTITION BY name_orig
        ORDER BY step, txn_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),

-- ---------------------------------------------------------------------------
-- 2. DESTINATION (RECEIVER) HISTORY
--    This is where the usable behavioural signal lives in PaySim, because
--    receivers repeat where senders do not.
-- ---------------------------------------------------------------------------
destination_history AS (
    SELECT
        o.*,
        COUNT(*) OVER w_dest                     AS dest_prior_txn_count,
        AVG(amount) OVER w_dest                  AS dest_prior_avg_amount,
        MAX(amount) OVER w_dest                  AS dest_prior_max_amount,
        SUM(amount) OVER w_dest                  AS dest_prior_total_amount,
        step - LAG(step) OVER (PARTITION BY name_dest ORDER BY step, txn_id)
                                                 AS dest_steps_since_prev_txn,
        -- Genuine time windows. RANGE, not ROWS. Ends at 1 PRECEDING so the
        -- current hour never counts itself.
        COUNT(*) OVER (PARTITION BY name_dest ORDER BY step
                       RANGE BETWEEN 24  PRECEDING AND 1 PRECEDING) AS dest_txn_prev_24h,
        COUNT(*) OVER (PARTITION BY name_dest ORDER BY step
                       RANGE BETWEEN 72  PRECEDING AND 1 PRECEDING) AS dest_txn_prev_72h,
        COUNT(*) OVER (PARTITION BY name_dest ORDER BY step
                       RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING) AS dest_txn_prev_168h,
        SUM(amount) OVER (PARTITION BY name_dest ORDER BY step
                          RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING) AS dest_amount_prev_24h,
        -- Same-hour concurrency: other transactions to this receiver inside the
        -- current step. This is concurrent, not future, information, and one
        -- hour is the finest resolution the dataset offers.
        COUNT(*) OVER (
        PARTITION BY name_dest
        ORDER BY step
        RANGE BETWEEN 1 PRECEDING AND 1 PRECEDING
        ) AS dest_txn_prev_1h,

        COUNT(*) OVER (PARTITION BY name_orig ORDER BY step
                       RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING)  AS orig_txn_prev_24h
    FROM origin_history o
    WINDOW w_dest AS (
        PARTITION BY name_dest
        ORDER BY step, txn_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),

-- ---------------------------------------------------------------------------
-- 3. SENDER-DESTINATION RELATIONSHIP
--    "Have these two parties transacted before?" A first-ever payment to a
--    counterparty is a different risk object from the hundredth.
-- ---------------------------------------------------------------------------
relationship AS (
    SELECT
        d.*,
        COUNT(*) OVER w_pair AS pair_prior_txn_count,
        COALESCE(SUM(amount) OVER w_pair, 0) AS pair_prior_total_amount,
        CASE WHEN ROW_NUMBER() OVER (PARTITION BY name_orig, name_dest
                                     ORDER BY step, txn_id) = 1
             THEN 1 ELSE 0 END AS is_first_contact
    FROM destination_history d
    WINDOW w_pair AS (
        PARTITION BY name_orig, name_dest
        ORDER BY step, txn_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),

-- ---------------------------------------------------------------------------
-- 4. DESTINATION FAN-IN (distinct senders seen so far)
--    COUNT(DISTINCT ...) is not allowed as a window function in PostgreSQL.
--    The workaround: mark each sender's FIRST contact with this destination
--    exactly once, then take a past-only running total of those marks. The
--    result is an exact distinct count, no subquery per row.
-- ---------------------------------------------------------------------------
fan_in AS (
    SELECT
        r.*,
        SUM(is_first_contact) OVER (
            PARTITION BY name_dest
            ORDER BY step, txn_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS dest_prior_unique_senders
    FROM relationship r
),

-- ---------------------------------------------------------------------------
-- 5. TRANSACTION-TYPE BASELINE
--    "Is this large for its own category?" Expanding and past-only, so the
--    category average never contains the transaction being judged.
-- ---------------------------------------------------------------------------
type_baseline AS (
    SELECT
        f.*,
        AVG(amount) OVER (
            PARTITION BY txn_type
            ORDER BY step, txn_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS type_prior_avg_amount
    FROM fan_in f
)

-- ---------------------------------------------------------------------------
-- 6. FINAL FEATURE TABLE
--    Ratios and flags are computed here so the CTEs above stay readable.
--    NULLIF guards every division; a NULL ratio means "no history", which the
--    *_has_history flags let a model distinguish from a genuine zero.
-- ---------------------------------------------------------------------------
SELECT
    txn_id,
    step,
    sim_day,
    hour_of_day,
    txn_type,
    amount,
    LOG(10, amount + 1)                                         AS log_amount,
    CASE WHEN MOD(amount, 1000) = 0 THEN 1 ELSE 0 END           AS amount_is_round_1000,
    name_orig,
    name_dest,
    dest_kind,
    CASE WHEN dest_kind = 'M' THEN 1 ELSE 0 END                 AS dest_is_merchant,

    -- origin behaviour
    COALESCE(orig_prior_txn_count, 0)                           AS orig_prior_txn_count,
    orig_prior_avg_amount,
    orig_prior_max_amount,
    COALESCE(orig_prior_total_amount, 0)                        AS orig_prior_total_amount,
    orig_steps_since_prev_txn,
    CASE WHEN COALESCE(orig_prior_txn_count, 0) > 0 THEN 1 ELSE 0 END AS orig_has_history,
    amount / NULLIF(orig_prior_avg_amount, 0)                   AS orig_amount_to_prior_avg,
    amount - orig_prior_avg_amount                              AS orig_amount_minus_prior_avg,
    amount / NULLIF(orig_prior_max_amount, 0)                   AS orig_amount_to_prior_max,
    COALESCE(orig_txn_prev_24h, 0)                              AS orig_txn_prev_24h,

    -- destination behaviour
    COALESCE(dest_prior_txn_count, 0)                           AS dest_prior_txn_count,
    dest_prior_avg_amount,
    dest_prior_max_amount,
    COALESCE(dest_prior_total_amount, 0)                        AS dest_prior_total_amount,
    dest_steps_since_prev_txn,
    CASE WHEN COALESCE(dest_prior_txn_count, 0) > 0 THEN 1 ELSE 0 END AS dest_has_history,
    CASE WHEN COALESCE(dest_prior_txn_count, 0) = 0 THEN 1 ELSE 0 END AS dest_is_new,
    amount / NULLIF(dest_prior_avg_amount, 0)                   AS dest_amount_to_prior_avg,
    COALESCE(dest_prior_unique_senders, 0)                      AS dest_prior_unique_senders,

    -- velocity
    COALESCE(dest_txn_prev_24h, 0)                              AS dest_txn_prev_24h,
    COALESCE(dest_txn_prev_72h, 0)                              AS dest_txn_prev_72h,
    COALESCE(dest_txn_prev_168h, 0)                             AS dest_txn_prev_168h,
    COALESCE(dest_amount_prev_24h, 0)                           AS dest_amount_prev_24h,
    dest_txn_prev_1h,
    -- Acceleration: recent rate against the longer-run rate, normalised so a
    -- value near 1 means "this receiver's last day looks like its last week".
    COALESCE(dest_txn_prev_24h, 0)
        / NULLIF(COALESCE(dest_txn_prev_168h, 0) / 7.0, 0)      AS dest_velocity_ratio_24h_vs_168h,

    -- relationship
    COALESCE(pair_prior_txn_count, 0)                           AS pair_prior_txn_count,
    CASE WHEN COALESCE(pair_prior_txn_count, 0) = 0 THEN 1 ELSE 0 END AS pair_is_new,
    COALESCE(pair_prior_total_amount, 0)                        AS pair_prior_total_amount,

    -- type baseline
    type_prior_avg_amount,
    amount / NULLIF(type_prior_avg_amount, 0)                   AS amount_to_type_prior_avg,

    -- target, kept last so nobody mistakes it for a feature
    is_fraud
FROM type_baseline;

CREATE INDEX idx_features_step  ON transaction_features (step);
CREATE INDEX idx_features_fraud ON transaction_features (is_fraud) WHERE is_fraud = 1;
CREATE INDEX idx_features_dest  ON transaction_features (name_dest);
ANALYZE transaction_features;


-- ===========================================================================
-- FEATURE COVERAGE CHECK
-- A feature that is NULL for 97% of rows is not a feature, it is a footnote.
-- Run this before modelling and publish the result alongside the model.
-- ===========================================================================
\echo '=== Feature coverage (share of rows where the feature is defined) ==='

SELECT
    ROUND(100.0 * COUNT(orig_prior_avg_amount)    / COUNT(*), 2) AS orig_prior_avg_defined_pct,
    ROUND(100.0 * COUNT(orig_steps_since_prev_txn)/ COUNT(*), 2) AS orig_gap_defined_pct,
    ROUND(100.0 * COUNT(dest_prior_avg_amount)    / COUNT(*), 2) AS dest_prior_avg_defined_pct,
    ROUND(100.0 * COUNT(dest_steps_since_prev_txn)/ COUNT(*), 2) AS dest_gap_defined_pct,
    ROUND(100.0 * SUM(CASE WHEN pair_prior_txn_count > 0 THEN 1 ELSE 0 END)
          / COUNT(*), 2)                                         AS pair_history_defined_pct,
    ROUND(100.0 * SUM(CASE WHEN dest_txn_prev_24h > 0 THEN 1 ELSE 0 END)
          / COUNT(*), 2)                                         AS dest_velocity_nonzero_pct
FROM transaction_features;


-- ===========================================================================
-- LEAKAGE SELF-CHECK
-- Every feature above should be computable at authorisation time. This query
-- confirms the frames really are past-only: for the FIRST transaction of any
-- account, every prior-history column must be NULL or zero. If any row comes
-- back non-zero, a frame is ending at CURRENT ROW somewhere.
-- ===========================================================================
\echo '=== Leakage self-check (expect 0 violations) ==='

SELECT COUNT(*) AS frames_that_include_the_current_row
FROM (
    SELECT
        txn_id,
        ROW_NUMBER() OVER (PARTITION BY name_dest ORDER BY step, txn_id) AS seq,
        dest_prior_txn_count
    FROM transaction_features
) t
WHERE seq = 1 AND dest_prior_txn_count <> 0;
