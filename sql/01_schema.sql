-- ===========================================================================
-- 01_schema.sql  |  Raw landing table for PaySim
--
-- Run:
--   createdb fraud_analytics
--   psql -d fraud_analytics -f sql/01_schema.sql
--
-- Then load the CSV (from psql, so the path is client-side):
--   \copy raw_transactions(step, txn_type, amount, name_orig, old_balance_orig,
--        new_balance_orig, name_dest, old_balance_dest, new_balance_dest,
--        is_fraud, is_flagged_fraud)
--   FROM 'data/raw/PS_20174392719_1491204439457_log.csv' CSV HEADER;
--
-- COLUMN NAMING
-- The CSV ships mixed-case names with an inconsistency that bites everyone:
-- "oldbalanceOrg" (no i) next to "newbalanceOrig" (with i). PostgreSQL folds
-- unquoted identifiers to lower case, so keeping the original spelling would
-- force double quotes on every reference forever. We rename once, here, and
-- document the mapping.
--
--   step            -> step                 (hour index 1..744, NOT a timestamp)
--   type            -> txn_type             ("type" is close to reserved usage)
--   amount          -> amount
--   nameOrig        -> name_orig
--   oldbalanceOrg   -> old_balance_orig
--   newbalanceOrig  -> new_balance_orig
--   nameDest        -> name_dest
--   oldbalanceDest  -> old_balance_dest
--   newbalanceDest  -> new_balance_dest
--   isFraud         -> is_fraud             (target)
--   isFlaggedFraud  -> is_flagged_fraud     (simulator control rule)
-- ===========================================================================

DROP TABLE IF EXISTS raw_transactions CASCADE;

CREATE TABLE raw_transactions (
    step                SMALLINT        NOT NULL,
    txn_type            VARCHAR(16)     NOT NULL,
    amount              NUMERIC(18, 2)  NOT NULL,
    name_orig           VARCHAR(16)     NOT NULL,
    old_balance_orig    NUMERIC(18, 2),
    new_balance_orig    NUMERIC(18, 2),
    name_dest           VARCHAR(16)     NOT NULL,
    old_balance_dest    NUMERIC(18, 2),
    new_balance_dest    NUMERIC(18, 2),
    is_fraud            SMALLINT        NOT NULL,
    is_flagged_fraud    SMALLINT        NOT NULL
);

-- NUMERIC rather than DOUBLE PRECISION for money: exact decimal arithmetic
-- matters for the balance-reconciliation checks in 02_data_validation.sql,
-- where floating point noise would masquerade as a data-quality finding.

COMMENT ON COLUMN raw_transactions.step IS
    'Hour index of the simulation, 1..744. Not a timestamp. No sub-hourly resolution exists.';
COMMENT ON COLUMN raw_transactions.is_fraud IS
    'Target variable: 1 if the transaction was part of a fraudulent takeover.';
COMMENT ON COLUMN raw_transactions.is_flagged_fraud IS
    'Simulator control rule output, not an observation. Target-derived: excluded from the model.';

-- ---------------------------------------------------------------------------
-- A surrogate key. The CSV has no transaction identifier, and window functions
-- need a deterministic tiebreak when several transactions share the same step.
-- ---------------------------------------------------------------------------
ALTER TABLE raw_transactions ADD COLUMN txn_id BIGSERIAL PRIMARY KEY;

-- ---------------------------------------------------------------------------
-- Indexes. Build them AFTER the COPY: loading 6.3M rows into an indexed table
-- is several times slower than loading then indexing.
-- ---------------------------------------------------------------------------
CREATE INDEX idx_raw_step        ON raw_transactions (step);
CREATE INDEX idx_raw_orig_step   ON raw_transactions (name_orig, step);
CREATE INDEX idx_raw_dest_step   ON raw_transactions (name_dest, step);
CREATE INDEX idx_raw_pair_step   ON raw_transactions (name_orig, name_dest, step);
CREATE INDEX idx_raw_type        ON raw_transactions (txn_type);
CREATE INDEX idx_raw_fraud       ON raw_transactions (is_fraud) WHERE is_fraud = 1;

ANALYZE raw_transactions;

-- ---------------------------------------------------------------------------
-- Sanity check. Expected on the full Kaggle file: 6,362,620 rows, 744 steps,
-- 5 transaction types. Verify against YOUR load rather than trusting the
-- numbers in this comment.
-- ---------------------------------------------------------------------------
SELECT
    COUNT(*)                      AS rows_loaded,
    COUNT(DISTINCT step)          AS distinct_steps,
    MIN(step)                     AS first_step,
    MAX(step)                     AS last_step,
    COUNT(DISTINCT txn_type)      AS transaction_types,
    SUM(is_fraud)                 AS fraud_rows,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM raw_transactions;
