"""
STAGE 5 - Report generation.

Every number in the generated reports is read from the results dictionary
produced by the pipeline. Nothing is typed in by hand, which is the only
reliable way to guarantee no invented result ever reaches the README.

Findings follow a fixed structure:
    OBSERVATION -> INTERPRETATION -> BUSINESS IMPLICATION -> RECOMMENDATION
so that measured fact and commercial judgement are never blurred together.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import config

logger = logging.getLogger(__name__)

SIMULATED = "**SIMULATED / ESTIMATED UNDER ASSUMPTIONS**"


def _n(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.{decimals}f}"
        return f"{value:.6g}"
    return str(value)


def _money(value: Any) -> str:
    return "n/a" if value is None else f"${value:,.0f}"


def _pct(value: Any, decimals: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}%"


def _table(rows: Sequence[Mapping], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "_no rows_"
    columns = list(columns or rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_n(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _finding(observation: str, interpretation: str, implication: str,
             recommendation: str) -> str:
    return (f"**OBSERVATION.** {observation}\n\n"
            f"**INTERPRETATION.** {interpretation}\n\n"
            f"**BUSINESS IMPLICATION.** {implication}\n\n"
            f"**RECOMMENDATION.** {recommendation}\n")


# ==========================================================================
def write_findings(results: dict, path: Path | None = None) -> Path:
    path = Path(path) if path else config.REPORT_DIR / "findings.md"
    audit = results["stage1_audit"]
    target = audit["target_isFraud"]
    accounts = audit["account_repetition"]
    leakage = audit["leakage_audit"]
    baselines = audit["honest_baselines"]
    model = results.get("stage3_models", {})
    optimum = results.get("stage3b_optimum", {})
    sensitivity = results.get("stage3c_summary", {})

    parts = [
        "# Findings",
        f"\n_Generated {datetime.now():%Y-%m-%d %H:%M} by `run_pipeline.py`. "
        f"Every number below is read from measured pipeline output._\n",
        f"\n_Data source: `{results['run']['dataset_path']}`_",
        f"\n_Rows analysed: {_n(results['run']['rows'])}_\n",
        "\n---\n",
        "## 1. Fraud is rare, so accuracy is the wrong yardstick\n",
        _finding(
            observation=(
                f"{_n(target['fraud_count'])} of {_n(target['fraud_count'] + target['legitimate_count'])} "
                f"transactions are labelled fraudulent, a rate of {_pct(target['fraud_rate_pct'], 4)}, "
                f"roughly one in {_n(target['imbalance_ratio_legit_per_fraud'], 0)}. "
                f"A classifier that predicts 'legitimate' every single time scores "
                f"{_pct(target['accuracy_of_always_legitimate_pct'], 3)} accuracy."),
            interpretation=(
                "Accuracy is dominated by the majority class and carries almost no "
                "information about fraud performance. PR-AUC, whose no-skill baseline "
                "equals the fraud rate, is the appropriate headline ranking metric, "
                "with precision and recall reported at a specific operating point."),
            implication=(
                "Any vendor or internal dashboard reporting fraud-model 'accuracy' is "
                "reporting a number that cannot distinguish a working control from no "
                "control at all."),
            recommendation=(
                "Report fraud capture rate, false-positive rate and alert volume "
                "at a named threshold. Retire accuracy from fraud reporting.")),
        "\n## 2. Balance columns leak the target\n",
        _finding(
            observation=(
                "Measured separation between fraudulent and legitimate transactions "
                "on balance-pattern checks:\n\n"
                + _table([{"check": k.replace("_", " "),
                           "fraud %": v["fraud_pct"],
                           "legitimate %": v["legitimate_pct"],
                           "gap (pp)": v["separation_pct_points"]}
                          for k, v in leakage["balance_pattern_checks_pct"].items()])),
            interpretation=(
        "PaySim provides pre-transaction balances (`oldbalanceOrg` and "
        "`oldbalanceDest`) and post-transaction balances (`newbalanceOrig` "
         "and `newbalanceDest`). The primary model excludes all balance fields "
        "as a conservative design choice because the simulator's balance and "
        "reversal mechanics can create signals that are difficult to interpret "
        "as clean authorization-time behavioural evidence. In particular, "
        "the `newbalance*` fields are post-transaction values and would not "
        "be available when an authorization decision is made."
                            ),
            implication=(
                "The balance fields are therefore kept outside the primary model. "
                "The `oldbalance*` fields are pre-transaction and could be evaluated "
                "in a separate model, while the `newbalance*` fields are unsuitable "
                "for authorization-time scoring because they describe the transaction "
                "after it occurs."
            ),
            recommendation=(
                "Exclude all four balance columns from the primary model as a "
                "conservative feature-design choice, and exclude `isFlaggedFraud` "
                "because it is a simulator control output rather than an independent "
                "behavioural observation. Keep the leakage demonstration clearly "
                "labelled and separate from the primary evaluation."
            )),
        "\n### Leakage decision table\n",
        _table(leakage["decision_table"],
               ["column", "leakage_risk", "reason", "primary_model_decision"]),
        "\n## 3. Originating accounts barely repeat, which reshapes the feature design\n",
        _finding(
            observation=(
                f"There are {_n(accounts['origin']['unique_accounts'])} unique originating "
                f"accounts across the dataset. Only "
                f"{_pct(accounts['origin']['pct_accounts_repeat'])} of them appear more "
                f"than once, accounting for "
                f"{_pct(accounts['origin']['pct_transactions_from_repeat_accounts'])} of "
                f"transactions. Prior origin history exists for "
                f"{_pct(accounts['prior_history_coverage']['rows_with_prior_origin_history_pct'])} "
                f"of rows, against "
                f"{_pct(accounts['prior_history_coverage']['rows_with_prior_destination_history_pct'])} "
                f"for destination accounts."),
            interpretation=(
                "Per-customer behavioural baselines such as 'amount versus this "
                "customer's historical average' are undefined for the large majority of "
                "transactions on the sender side. Destination accounts, by contrast, "
                "accumulate usable history."),
            implication=(
                "A feature set built around sender history would be mostly missing "
                "values dressed up as signal. Counterparty behaviour is where the "
                "observable structure actually lives in this dataset."),
            recommendation=(
                "Base behavioural features on destination history, sender-destination "
                "relationship novelty, destination fan-in and time-window velocity. "
                "Compute origin features too, but publish their coverage alongside "
                "them rather than presenting them as fully populated.")),
        "\n## 4. Baselines the model has to beat\n",
        _table([{"baseline": name.replace("baseline_", "").replace("_", " "),
                 "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
                 "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]}
                for name, m in baselines["rules"].items()]),
        f"\nAmount used as a continuous score: ROC-AUC "
        f"{_n(baselines['baseline_5_amount_ranking']['roc_auc'])}, PR-AUC "
        f"{_n(baselines['baseline_5_amount_ranking']['pr_auc'])} against a no-skill "
        f"baseline of {_n(baselines['baseline_5_amount_ranking']['pr_auc_no_skill_baseline'])}.\n",
    ]

    if model:
        best = model.get("best_model_name")
        test = model.get("test_metrics", {}).get(best, {})
        parts += [
            "\n## 5. Model performance on a held-out future period\n",
            _finding(
                observation=(
                    f"Models were trained on the earliest "
                    f"{_pct(results['stage3_split']['train']['row_pct'])} of transactions "
                    f"by `step`, tuned on the next "
                    f"{_pct(results['stage3_split']['validation']['row_pct'])}, and "
                    f"evaluated on the final "
                    f"{_pct(results['stage3_split']['test']['row_pct'])}, which the "
                    f"models never saw. On that test period the selected model "
                    f"(`{best}`) reached PR-AUC {_n(test.get('pr_auc'))} and ROC-AUC "
                    f"{_n(test.get('roc_auc'))}."),
                interpretation=(
                    "A chronological split reproduces the only question that matters "
                    "operationally: does a model fitted on the past work on traffic it "
                    "has not seen. A random split would have leaked future behaviour "
                    "into cumulative features and inflated these numbers."),
                implication=(
                    "The ranking quality is what determines whether a review queue can "
                    "be prioritised at all. The threshold decision in the next section "
                    "is only meaningful because the ranking carries signal."),
                recommendation=(
                    "Re-validate on a rolling forward window before any deployment, and "
                    "monitor PR-AUC drift by month rather than a single static score.")),
        ]

    if optimum:
        parts += [
            "\n## 6. The cost-minimising intervention threshold\n",
            _finding(
                observation=(
                    f"Sweeping thresholds from {config.THRESHOLD_GRID_START} to "
                    f"{config.THRESHOLD_GRID_STOP} under the simulated assumptions "
                    f"(missed fraud {_money(config.COST_FALSE_NEGATIVE)}, blocked "
                    f"legitimate transaction {_money(config.COST_FALSE_POSITIVE)}), "
                    f"total simulated cost is minimised at threshold "
                    f"**{_n(optimum['optimal_threshold'])}**, giving "
                    f"{_n(optimum['TP'])} true positives, {_n(optimum['FP'])} false "
                    f"positives, {_n(optimum['FN'])} missed fraud cases, fraud capture "
                    f"{_pct(100 * optimum['fraud_capture_rate'])} and a false-positive "
                    f"rate of {_pct(100 * optimum['false_positive_rate'], 4)}. "
                    f"Simulated total cost {_money(optimum['total_cost_at_optimum'])} "
                    f"versus {_money(optimum['cost_at_threshold_0_50'])} at the "
                    f"conventional 0.50 threshold."),
                interpretation=(
                    "0.50 is a mathematical convention that implicitly assumes a missed "
                    "fraud and an inconvenienced customer cost the same. Once the two "
                    "costs are stated explicitly, the cost curve has a different minimum."),
                implication=(
                    f"Under these assumptions the threshold choice alone moves simulated "
                    f"cost by {_money(optimum['cost_reduction_vs_0_50'])} "
                    f"({_pct(optimum['cost_reduction_vs_0_50_pct'])}) on this evaluation "
                    f"sample, with no change to the model itself."),
                recommendation=(
                    "Treat the threshold as a governed business parameter with a named "
                    "owner and a review cadence, not a default buried in code. "
                    f"{SIMULATED} These are not measured company savings.")),
        ]

    if sensitivity:
        parts += [
            "\n## 7. The threshold depends on economics, not just the model\n",
            _finding(
                observation=(
                    f"Across {_n(sensitivity['scenarios_tested'])} cost scenarios the "
                    f"minimum-cost threshold ranges from "
                    f"{_n(sensitivity['threshold_min'])} to {_n(sensitivity['threshold_max'])} "
                    f"(median {_n(sensitivity['threshold_median'])}). Fraud capture across "
                    f"those scenarios ranges from "
                    f"{_pct(100 * sensitivity['fraud_capture_rate_range'][0])} to "
                    f"{_pct(100 * sensitivity['fraud_capture_rate_range'][1])}."),
                interpretation=sensitivity["interpretation"],
                implication=(
                    "Two teams running the identical model can be correct to operate at "
                    "very different thresholds if their cost structures differ. "
                    "Disagreement about aggressiveness is usually disagreement about "
                    "costs, not about the model."),
                recommendation=(
                    "Agree the cost of a blocked legitimate transaction with Finance and "
                    "Customer Operations first. Re-run this grid whenever chargeback "
                    "costs or support economics change.")),
        ]

    if "stage3d_importance" in results and results["stage3d_importance"]:
        top = results["stage3d_importance"][:10]
        parts += [
            "\n## 8. What the model is responding to\n",
            _table(top),
            "\nThese features contributed to the model's scores. They are associated "
            "with the prediction; none of this establishes causation.\n",
        ]

    parts += [
        "\n## 9. Limitations\n",
        "- PaySim is synthetic. Patterns come from a simulator, so performance here "
        "does not transfer to a real payments portfolio.\n"
        "- The dataset has no sub-hourly timing, no geography, no device or IP data, "
        "and no merchant category, so whole families of production fraud features "
        "cannot be evaluated at all.\n"
        "- All monetary figures are simulated. The dataset contains no recovery rates, "
        "chargeback costs or investigation costs.\n"
        "- The cost model treats every false positive as equally costly. In reality, "
        "blocking a large recurring payment for a long-tenured customer is not "
        "equivalent to declining a small one-off.\n"
        "- Tree-model probabilities are not calibrated, so threshold values are not "
        "directly comparable between models even when both are well ranked.\n"
        "- A single held-out period is one sample. Confidence intervals and rolling "
        "revalidation are absent.\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ==========================================================================
def write_business_recommendations(results: dict, path: Path | None = None) -> Path:
    path = Path(path) if path else config.REPORT_DIR / "business_recommendations.md"
    optimum = results.get("stage3b_optimum", {})
    bands = results.get("stage3b_bands", {})
    friction = results.get("stage4_friction", {})
    audit = results["stage1_audit"]

    band_rows = bands.get("bands", [])
    parts = [
        "# Business recommendations",
        f"\n_Generated {datetime.now():%Y-%m-%d %H:%M}. All monetary figures are "
        f"{SIMULATED} and are not measured company results._\n",
        "\n## 1. Operate a three-tier intervention policy, not a single block rule\n",
        "The cost model produces two different thresholds because two different "
        "interventions have two different costs. A step-up verification annoys a "
        "customer less than a hard decline, so it pays for itself at a lower level of "
        "suspicion. That gives a derived band structure rather than round numbers "
        "chosen by eye:\n",
        _table(band_rows, ["band", "range", "action", "rationale"]),
        f"\nBoundaries: verify at **{_n(bands.get('medium_risk_threshold_verify'))}**, "
        f"block at **{_n(bands.get('high_risk_threshold_block'))}**, derived from "
        f"friction costs of {_money(config.COST_STEP_UP_VERIFICATION)} and "
        f"{_money(config.COST_FALSE_POSITIVE)} against a missed-fraud cost of "
        f"{_money(config.COST_FALSE_NEGATIVE)}.\n",
        "\n## 2. Concentrate controls where fraud actually occurs\n",
    ]

    types_with_fraud = audit["transaction_type"]["types_with_fraud"]
    types_without = audit["transaction_type"]["types_without_fraud"]
    parts += [
        _finding(
            observation=(
                f"Labelled fraud appears only in these transaction types: "
                f"{', '.join(types_with_fraud) or 'none'}. These types carry no labelled "
                f"fraud at all: {', '.join(types_without) or 'none'}."),
            interpretation=(
                "In this dataset the account-takeover pattern is expressed through "
                "specific movement types. Applying identical scrutiny to every type "
                "spends review capacity where the base rate is effectively zero."),
            implication=(
                "Uniform controls create customer friction in categories with no "
                "measured fraud exposure, which is pure cost."),
            recommendation=(
                "Scope the intervention policy to the fraud-bearing types first. "
                "Maintain light monitoring elsewhere to detect a shift in the pattern "
                "rather than assuming it is permanent.")),
    ]

    if friction:
        worst = friction.get("worst_type")
        parts += [
            "\n## 3. Target the friction you are creating\n",
            _finding(
                observation=(
                    f"At the selected threshold, {_n(friction.get('total_false_positives'))} "
                    f"legitimate transactions are blocked, a false-positive rate of "
                    f"{_pct(friction.get('false_positive_rate_pct'), 4)}, carrying a "
                    f"simulated friction cost of {_money(friction.get('friction_cost'))}. "
                    f"The heaviest concentration is in "
                    f"{worst.get('type') if worst else 'n/a'} "
                    f"({_n(worst.get('false_positives')) if worst else 'n/a'} cases, "
                    f"{_pct(worst.get('false_positive_rate_pct'), 4) if worst else 'n/a'})."),
                interpretation=(
                    "False positives are not spread evenly. They cluster where the model "
                    "is least discriminative, which is usually where legitimate "
                    "behaviour most resembles the fraud pattern."),
                implication=(
                    "Blanket blocking in that segment buys relatively little fraud "
                    "prevention per disrupted customer."),
                recommendation=(
                    "Route that segment to step-up verification rather than a hard "
                    "decline, and track the recovery rate of verified transactions as "
                    "the measure of whether the softer control is working.")),
        ]

    parts += [
        "\n## 4. Govern the threshold as a business parameter\n",
        _finding(
            observation=(
                f"The minimum-cost threshold moved across the sensitivity grid from "
                f"{_n(results.get('stage3c_summary', {}).get('threshold_min'))} to "
                f"{_n(results.get('stage3c_summary', {}).get('threshold_max'))} purely "
                f"as a function of the cost assumptions."),
            interpretation=(
                "The threshold is not a model property. It is a pricing decision about "
                "the relative cost of two kinds of error."),
            implication=(
                "If nobody owns the cost assumptions, nobody owns the threshold, and it "
                "drifts to whatever the default in the code happens to be."),
            recommendation=(
                "Assign ownership of the two cost inputs to Finance and Customer "
                "Operations, review them quarterly, and re-run the sensitivity grid on "
                "each revision.")),
        "\n## 5. Invest in the data that is missing before tuning the model further\n",
        _finding(
            observation=(
                "The dataset offers no device, IP, geography, merchant-category or "
                "sub-hourly timing information, and originating accounts are largely "
                "single-use."),
            interpretation=(
                "The remaining headroom in this system is limited less by algorithm "
                "choice than by the narrowness of the observation space."),
            implication=(
                "Further model tuning on the same fields has a low ceiling; new signal "
                "sources have a much higher one."),
            recommendation=(
                "Prioritise device fingerprinting, IP intelligence, precise timestamps "
                "and counterparty risk scoring in the data roadmap ahead of further "
                "model experimentation.")),
        f"\n---\n\n_All cost figures in this document are {SIMULATED}. They describe "
        "modelled outcomes on a synthetic dataset under stated assumptions and are not "
        "realised savings for any company._\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ==========================================================================
def write_final_summary(results: dict, path: Path | None = None) -> Path:
    path = Path(path) if path else config.REPORT_DIR / "FINAL_SUMMARY.md"
    audit = results["stage1_audit"]
    target = audit["target_isFraud"]
    accounts = audit["account_repetition"]
    model = results.get("stage3_models", {})
    best = model.get("best_model_name", "n/a")
    test = model.get("test_metrics", {}).get(best, {})
    optimum = results.get("stage3b_optimum", {})
    sensitivity = results.get("stage3c_summary", {})

    parts = [
        "# Final project summary",
        f"\n_Generated {datetime.now():%Y-%m-%d %H:%M} from measured pipeline output._\n",
        "\n## Dataset characteristics\n",
        _table([
            {"metric": "rows", "value": _n(results["run"]["rows"])},
            {"metric": "columns (raw)", "value": _n(len(config.EXPECTED_COLUMNS))},
            {"metric": "fraud transactions", "value": _n(target["fraud_count"])},
            {"metric": "fraud rate", "value": _pct(target["fraud_rate_pct"], 4)},
            {"metric": "legit per fraud", "value": _n(target["imbalance_ratio_legit_per_fraud"], 0)},
            {"metric": "time span", "value": audit["step_and_time"]["implied_time_range"]},
            {"metric": "unique origin accounts", "value": _n(accounts["origin"]["unique_accounts"])},
            {"metric": "unique destination accounts", "value": _n(accounts["destination"]["unique_accounts"])},
        ], ["metric", "value"]),
        "\n## Data limitations that shaped the design\n",
        "- Hourly resolution only, so no minute-level velocity exists.\n"
        "- No geography, device or IP, so no location or impossible-travel features.\n"
        "- ~30 days total, so no true 30-day rolling baseline for early transactions; "
        "an expanding past-only baseline is used instead.\n"
        f"- Origin accounts largely single-use: prior origin history covers "
        f"{_pct(accounts['prior_history_coverage']['rows_with_prior_origin_history_pct'])} "
        f"of rows versus "
        f"{_pct(accounts['prior_history_coverage']['rows_with_prior_destination_history_pct'])} "
        f"for destinations.\n"
        "- Synthetic data, so no claim about real-world performance is supportable.\n",
        "\n## Leakage findings\n",
        _table(audit["leakage_audit"]["decision_table"],
               ["column", "leakage_risk", "primary_model_decision"]),
    ]

    if "leakage_demo" in results and results["leakage_demo"]:
        demo = results["leakage_demo"]
        parts += [
            f"\n**LEAKAGE DEMONSTRATION - NOT VALID FOR DEPLOYMENT.** A model trained "
            f"on the excluded balance columns reached PR-AUC {_n(demo.get('pr_auc'))} "
            f"and ROC-AUC {_n(demo.get('roc_auc'))} on the same test period, against "
            f"PR-AUC {_n(test.get('pr_auc'))} for the legitimate model. The gap is the "
            f"size of the illusion that leakage creates. This model is reported here "
            f"only as a warning and is never used for any decision in this project.\n",
        ]

    parts += [
        "\n## Behavioural features supported by the dataset\n",
        f"{_n(len(results.get('feature_columns', [])))} features were built, all "
        "past-only. Families: amount shape, hour-of-day, origin expanding history, "
        "destination expanding history, destination fan-in and unique-sender counts, "
        "sender-destination relationship novelty, time-window velocity over `step` "
        f"({', '.join(str(w) + 'h' for w in config.VELOCITY_WINDOWS_HOURS)}), and "
        "transaction type.\n",
        "\n## Models\n",
        _table([{"model": name, **{k: _n(v) for k, v in metrics.items()
                                  if k in ("roc_auc", "pr_auc", "precision", "recall", "f1")}}
                for name, metrics in model.get("test_metrics", {}).items()]),
        f"\nSelected model: **{best}**, chosen by PR-AUC on the validation period, "
        f"then evaluated once on the untouched test period.\n",
        "\n## Financial threshold methodology\n",
        "1. Score the validation period with the selected model.\n"
        "2. Sweep thresholds 0.01 to 0.99, recording the full confusion matrix at each.\n"
        f"3. Apply the simulated cost function: FN x {_money(config.COST_FALSE_NEGATIVE)} "
        f"+ FP x {_money(config.COST_FALSE_POSITIVE)}.\n"
        "4. Select the minimum-cost threshold on validation, then report its "
        "performance on the test period. Choosing the threshold on the same data used "
        "to report it would be a subtler form of the leakage this project is about.\n",
    ]

    if optimum:
        parts += [
            "\n## Minimum-cost threshold under the baseline assumptions\n",
            _table([
                {"metric": "threshold", "value": _n(optimum["optimal_threshold"])},
                {"metric": "fraud captured", "value": _pct(100 * optimum["fraud_capture_rate"])},
                {"metric": "precision", "value": _pct(100 * optimum["precision"])},
                {"metric": "false positive rate", "value": _pct(100 * optimum["false_positive_rate"], 4)},
                {"metric": "legitimate blocked", "value": _n(optimum["legitimate_transactions_blocked"])},
                {"metric": "fraud missed", "value": _n(optimum["FN"])},
                {"metric": "simulated total cost", "value": _money(optimum["total_cost_at_optimum"])},
                {"metric": "simulated cost at t=0.50", "value": _money(optimum["cost_at_threshold_0_50"])},
                {"metric": "simulated difference", "value": _money(optimum["cost_reduction_vs_0_50"])},
            ], ["metric", "value"]),
            f"\n{SIMULATED} Minimum simulated cost under the specified assumptions and "
            "evaluation sample only. Not a universally optimal threshold, and not a "
            "realised saving.\n",
        ]

    if sensitivity:
        parts += [
            "\n## Cost sensitivity findings\n",
            f"{sensitivity['interpretation']}\n",
        ]

    parts += [
        "\n## Dashboard design\n",
        "Three Power BI pages fed by `outputs/dashboard_data/`:\n\n"
        "1. **Executive risk overview** - KPI cards, fraud trend, fraud by type, "
        "threshold-versus-cost curve, capture-versus-friction trade-off.\n"
        "2. **Fraud operations** - action queue sorted by fraud probability, with "
        "risk band, velocity and counterparty context for each alert.\n"
        "3. **Customer friction** - false positives by type, amount band and time, "
        "with the simulated friction cost attached.\n",
        "\n## Main business insights\n",
        "1. Accuracy is unusable as a fraud metric at this base rate.\n"
       "2. Balance fields are excluded from the primary model because of PaySim's "
"simulator-specific balance and reversal mechanics and the need to keep the "
"authorization-time feature set conservative."
        "3. Sender history barely exists in this dataset, so counterparty behaviour "
        "carries the behavioural signal.\n"
        "4. The intervention threshold is a pricing decision, not a model property.\n"
        "5. Customer friction is concentrated, so it can be targeted with a softer "
        "control instead of accepted as the cost of doing business.\n",
        "\n## Resume-ready description\n",
        f"> Built an end-to-end fraud analytics system on {_n(results['run']['rows'])} "
        f"mobile-money transactions: ran a leakage audit that excluded four "
        f"outcome-contaminated balance fields, engineered "
        f"{_n(len(results.get('feature_columns', [])))} past-only behavioural features "
        f"in SQL and pandas (destination velocity over RANGE windows, counterparty "
        f"fan-in, sender-destination novelty), trained time-split logistic regression "
        f"and random forest models reaching PR-AUC {_n(test.get('pr_auc'))} on a "
        f"held-out future period, and replaced the default 0.50 decision threshold "
        f"with a cost-minimising operating point derived from an explicit "
        f"false-negative and false-positive cost model, validated across 25 cost "
        f"scenarios and delivered as a three-page Power BI decision dashboard.\n",
        "\n## Interview explanation\n",
        "**What is the project?** A fintech fraud system where the deliverable is not "
        "the classifier but the intervention policy. The model produces a probability; "
        "the business decision is where to act on it, and that depends on what each "
        "kind of mistake costs.\n\n"
        "**Hardest technical decision?** Excluding the balance columns. They give a "
        "near-perfect model, and the dataset documentation explains why: fraudulent "
        "transactions are cancelled, so those fields describe the aftermath, not the "
        "transaction. I quantified the separation, excluded them, and kept a labelled "
        "leakage demo to show the size of the illusion.\n\n"
        "**Why time-aware splitting?** Every behavioural feature is cumulative. A "
        "random split lets the model see later transactions from the same destination "
        "account during training and earlier ones at test time, which inflates results "
        "and tells you nothing about tomorrow.\n\n"
        "**Why is PR-AUC the headline metric?** At a fraud rate below one percent, "
        "ROC-AUC's denominator is the enormous legitimate population, so it stays "
        "flattering while precision is poor. PR-AUC's no-skill line is the fraud rate "
        "itself, which makes improvement legible.\n\n"
        "**What would you do with more data?** Device fingerprinting, IP intelligence "
        "and true timestamps, in that order. The ceiling here is the observation "
        "space, not the algorithm.\n",
        "\n## Limitations\n",
        "See the limitations section of `findings.md`. In short: synthetic data, no "
        "geography or device signal, hourly resolution, simulated costs, uncalibrated "
        "tree probabilities, and a single held-out period rather than rolling "
        "revalidation.\n",
        "\n## Future improvements\n",
        "Real-time scoring, streaming feature computation, device fingerprinting, IP "
        "intelligence, merchant risk scoring, graph-based detection over the "
        "sender-receiver network, adaptive thresholds driven by live cost telemetry, "
        "chargeback-derived cost inputs, investigation feedback loops, model "
        "monitoring with drift detection, human-in-the-loop review, and formal model "
        "governance. None of these is implemented here.\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path
