"""
End-to-end pipeline: Stage 1 through Stage 5 in one command.

    python run_pipeline.py                      # full dataset, full models
    python run_pipeline.py --fast               # lighter models, quicker
    python run_pipeline.py --nrows 1000000      # development subset
    python run_pipeline.py --help

Outputs:
    outputs/model_results/    JSON results, feature coverage, sweeps
    outputs/figures/          every chart
    outputs/dashboard_data/   Power BI ready CSVs
    reports/                  findings.md, business_recommendations.md,
                              FINAL_SUMMARY.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src import (
    config,
    dashboard_export,
    data_profiling,
    evaluate_models,
    explainability,
    feature_engineering,
    plots,
    preprocessing,
    report_generator,
    sensitivity_analysis,
    threshold_optimization,
    train_models,
)
from src.utils import save_json, setup_logging, timed

log = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fintech fraud optimization pipeline")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Path to the PaySim CSV (default: auto-detect in data/raw)")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Read only the first N rows. PaySim is time-ordered, so "
                             "this is an early time window, not a random sample.")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--fast", action="store_true",
                        help="Lighter model settings for a quick first pass")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="Class re-weighting during training. Default 'none' keeps "
                             "probabilities calibrated, which the cost model relies on.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-leakage-demo", action="store_true",
                        help="Skip the labelled leakage demonstration model")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    config.ensure_directories()
    np.random.seed(config.RANDOM_SEED)

    results: dict = {"run": {}}

    # ======================================================================
    # LOAD
    # ======================================================================
    with timed("load and prepare"):
        try:
            path = args.csv or preprocessing.find_dataset_file()
            df = preprocessing.load_paysim(path, nrows=args.nrows, chunksize=args.chunksize)
        except (FileNotFoundError, preprocessing.DatasetContractError) as exc:
            log.error("%s", exc)
            return 1
        df = preprocessing.prepare(df)
        results["run"] = {
            "dataset_path": str(path),
            "rows": int(len(df)),
            "nrows_limit": args.nrows,
            "fast_mode": bool(args.fast),
            "class_weight": args.class_weight,
            "random_seed": config.RANDOM_SEED,
        }

    # ======================================================================
    # STAGE 1 - audit
    # ======================================================================
    with timed("STAGE 1 audit"):
        audit = data_profiling.run_full_audit(df)
        results["stage1_audit"] = audit
        save_json(audit, config.MODEL_RESULT_DIR / "stage1_audit.json")

    if not args.skip_plots:
        with timed("STAGE 1 figures"):
            plots.plot_class_imbalance(df)
            plots.plot_amount_distribution(df)
            plots.plot_fraud_by_type(df)
            plots.plot_hourly_pattern(df)
            plots.plot_account_repetition(df)
            plots.plot_leakage_separation(audit["leakage_audit"])

    # ======================================================================
    # STAGE 2 - behavioural features
    # ======================================================================
    with timed("STAGE 2 feature engineering"):
        featured, feature_columns = feature_engineering.build_features(df)
        coverage = feature_engineering.feature_coverage(featured, feature_columns)
        coverage.to_csv(config.MODEL_RESULT_DIR / "feature_coverage.csv", index=False)
        featured = feature_engineering.fill_missing(featured, feature_columns)
        results["feature_columns"] = feature_columns
        results["feature_coverage"] = coverage.to_dict("records")
        log.info("Features with <10%% coverage: %s",
                 coverage.loc[coverage["defined_pct"] < 10, "feature"].tolist())
        del df

    # ======================================================================
    # STAGE 3 - time-aware split and training
    # ======================================================================
    with timed("STAGE 3 split"):
        split = train_models.time_aware_split(featured)
        masks, summary = split["masks"], split["summary"]
        results["stage3_split"] = summary
        save_json(summary, config.MODEL_RESULT_DIR / "split_summary.json")

    y = featured[config.TARGET].to_numpy()
    X = featured[feature_columns]
    sets = {name: (X[mask], y[mask.to_numpy()]) for name, mask in masks.items()}

    if sets["validation"][1].sum() == 0 or sets["test"][1].sum() == 0:
        log.error("Validation or test split contains no fraud cases. "
                  "Increase --nrows or adjust the split fractions in src/config.py.")
        return 1

    with timed("STAGE 3 training"):
        class_weight = None if args.class_weight == "none" else "balanced"
        models = train_models.build_models(fast=args.fast, class_weight=class_weight)
        fitted = train_models.fit_models(models, *sets["train"])

    with timed("STAGE 3 evaluation"):
        val_scores, test_scores = {}, {}
        for name, model in fitted.items():
            val_scores[name] = train_models.predict_proba(model, sets["validation"][0])
            test_scores[name] = train_models.predict_proba(model, sets["test"][0])

        val_comparison = evaluate_models.compare_models(sets["validation"][1], val_scores)
        test_comparison = evaluate_models.compare_models(sets["test"][1], test_scores)
        val_comparison.to_csv(config.MODEL_RESULT_DIR / "model_comparison_validation.csv", index=False)
        test_comparison.to_csv(config.MODEL_RESULT_DIR / "model_comparison_test.csv", index=False)

        # Model selection happens on validation; test is touched once, at the end.
        best_name = str(val_comparison.iloc[0]["model"])
        log.info("Selected model by validation PR-AUC: %s", best_name)

        results["stage3_models"] = {
            "best_model_name": best_name,
            "selection_metric": "PR-AUC on the validation period",
            "validation_metrics": {
                name: evaluate_models.ranking_metrics(sets["validation"][1], scores)
                for name, scores in val_scores.items()},
            "test_metrics": {
                name: evaluate_models.ranking_metrics(sets["test"][1], scores)
                for name, scores in test_scores.items()},
            "top_k_capture_test": evaluate_models.top_k_capture(
                sets["test"][1], test_scores[best_name]),
            "xgboost_available": train_models.XGBOOST_AVAILABLE,
        }

    if not args.skip_plots:
        curves = {name: evaluate_models.curve_points(sets["test"][1], scores)
                  for name, scores in test_scores.items()}
        plots.plot_model_curves(curves, float(sets["test"][1].mean()))

    # ---- leakage demonstration (clearly labelled, never used for decisions)
    if not args.skip_leakage_demo:
        with timed("leakage demonstration model"):
            try:
                leak_cols = [c for c in config.LEAKAGE_SUSPECT_COLUMNS if c in featured.columns]
                X_leak = featured[leak_cols].astype("float32")
                demo = train_models.train_leakage_demo(
                    X_leak[masks["train"]], sets["train"][1], fast=args.fast)
                demo_scores = demo.predict_proba(X_leak[masks["test"]])[:, 1]
                results["leakage_demo"] = {
                    "label": "LEAKAGE DEMONSTRATION - NOT VALID FOR DEPLOYMENT",
                    "columns_used": leak_cols,
                    **evaluate_models.ranking_metrics(sets["test"][1], demo_scores),
                }
                log.warning("LEAKAGE DEMO (not deployable): PR-AUC %.4f vs legitimate "
                            "model PR-AUC %.4f",
                            results["leakage_demo"]["pr_auc"],
                            results["stage3_models"]["test_metrics"][best_name]["pr_auc"])
            except Exception:
                log.exception("Leakage demonstration failed; continuing.")

    # ======================================================================
    # STAGE 3B - threshold optimization
    # ======================================================================
    with timed("STAGE 3B threshold optimization"):
        val_sweep = threshold_optimization.threshold_sweep(
            sets["validation"][1], val_scores[best_name])
        chosen = threshold_optimization.optimal_threshold(val_sweep)
        threshold = chosen["optimal_threshold"]

        test_sweep = threshold_optimization.threshold_sweep(
            sets["test"][1], test_scores[best_name])
        optimum = threshold_optimization.summarise_at(test_sweep, threshold)
        optimum["threshold_selected_on"] = "validation period"
        optimum["threshold_reported_on"] = "test period"
        optimum["validation_optimum"] = chosen

        bands = threshold_optimization.derive_risk_bands(
            sets["validation"][1], val_scores[best_name])

        val_sweep.to_csv(config.MODEL_RESULT_DIR / "threshold_sweep_validation.csv", index=False)
        test_sweep.to_csv(config.MODEL_RESULT_DIR / "threshold_sweep_test.csv", index=False)
        results["stage3b_optimum"] = optimum
        results["stage3b_bands"] = bands
        log.info("Threshold chosen on validation: %.2f | simulated test cost %s",
                 threshold, f"${optimum['total_cost_at_optimum']:,.0f}")

    if not args.skip_plots:
        plots.plot_threshold_cost(test_sweep, optimum)

    # ======================================================================
    # STAGE 3C - cost sensitivity
    # ======================================================================
    with timed("STAGE 3C sensitivity"):
        grid = sensitivity_analysis.sensitivity_grid(sets["test"][1], test_scores[best_name])
        grid.to_csv(config.MODEL_RESULT_DIR / "cost_sensitivity_grid.csv", index=False)
        results["stage3c_grid"] = grid.to_dict("records")
        results["stage3c_summary"] = sensitivity_analysis.summarise_sensitivity(grid)

    if not args.skip_plots:
        plots.plot_sensitivity_heatmaps(grid)

    # ======================================================================
    # STAGE 3D - interpretability
    # ======================================================================
    with timed("STAGE 3D interpretability"):
        importance = explainability.model_feature_importance(fitted[best_name], feature_columns)
        if importance is not None:
            importance.to_csv(config.MODEL_RESULT_DIR / "feature_importance.csv", index=False)
            results["stage3d_importance"] = importance.head(25).to_dict("records")
            if not args.skip_plots:
                plots.plot_feature_importance(importance)

        permutation = explainability.permutation_feature_importance(
            fitted[best_name], sets["test"][0], sets["test"][1])
        if permutation is not None:
            permutation.to_csv(config.MODEL_RESULT_DIR / "permutation_importance.csv", index=False)
            results["stage3d_permutation"] = permutation.head(25).to_dict("records")
            if not args.skip_plots:
                plots.plot_feature_importance(
                    permutation, name="10b_permutation_importance.png")

        shap_table = explainability.shap_summary(fitted[best_name], sets["test"][0])
        if shap_table is not None:
            shap_table.to_csv(config.MODEL_RESULT_DIR / "shap_importance.csv", index=False)
            results["stage3d_shap"] = shap_table.head(25).to_dict("records")

    # ======================================================================
    # STAGE 4 - dashboard preparation
    # ======================================================================
    with timed("STAGE 4 dashboard export"):
        test_frame = featured.loc[masks["test"]].copy()
        risk_bands = threshold_optimization.assign_risk_band(test_scores[best_name], bands)
        scored = dashboard_export.build_scored_table(
            test_frame, test_scores[best_name], risk_bands, threshold)

        exported = dashboard_export.export_all(
            scored, test_sweep, grid, optimum, bands,
            results["stage3_models"]["test_metrics"][best_name])
        results["stage4_files"] = {k: str(v) for k, v in exported.items()}

        # Friction summary for the business report.
        legitimate = scored[scored[config.TARGET] == 0]
        false_positives = legitimate["outcome_class"] == "false_positive"
        by_type = (legitimate.assign(fp=false_positives)
                   .groupby("type", observed=True)["fp"].agg(["sum", "mean"])
                   .rename(columns={"sum": "false_positives", "mean": "rate"}))
        worst = by_type.sort_values("false_positives", ascending=False).head(1)
        results["stage4_friction"] = {
            "total_false_positives": int(false_positives.sum()),
            "false_positive_rate_pct": float(100 * false_positives.mean()),
            "friction_cost": float(false_positives.sum() * config.COST_FALSE_POSITIVE),
            "worst_type": ({"type": str(worst.index[0]),
                            "false_positives": int(worst["false_positives"].iloc[0]),
                            "false_positive_rate_pct": float(100 * worst["rate"].iloc[0])}
                           if len(worst) else None),
            "cost_basis": "SIMULATED / ESTIMATED UNDER ASSUMPTIONS",
        }

    if not args.skip_plots:
        plots.plot_false_positive_profile(scored)
        plots.plot_risk_band_distribution(scored)

    # ======================================================================
    # STAGE 5 - reports
    # ======================================================================
    with timed("STAGE 5 reports"):
        save_json(results, config.MODEL_RESULT_DIR / "pipeline_results.json")
        report_generator.write_findings(results)
        report_generator.write_business_recommendations(results)
        report_generator.write_final_summary(results)

    _print_summary(results)
    return 0


def _print_summary(results: dict) -> None:
    audit = results["stage1_audit"]
    target = audit["target_isFraud"]
    models = results.get("stage3_models", {})
    best = models.get("best_model_name", "n/a")
    test = models.get("test_metrics", {}).get(best, {})
    optimum = results.get("stage3b_optimum", {})
    sens = results.get("stage3c_summary", {})

    line = "=" * 74
    print(f"\n{line}\nPIPELINE SUMMARY\n{line}")
    print(f"Rows analysed              : {results['run']['rows']:,}")
    print(f"Fraud transactions         : {target['fraud_count']:,} "
          f"({target['fraud_rate_pct']:.4f}%)")
    print(f"'Always legitimate' accuracy: {target['accuracy_of_always_legitimate_pct']:.4f}%")
    print(f"Features built             : {len(results.get('feature_columns', []))}")
    print(f"Selected model             : {best} (by validation PR-AUC)")
    if test:
        print(f"Test PR-AUC / ROC-AUC      : {test.get('pr_auc'):.4f} / {test.get('roc_auc'):.4f}")
        print(f"PR-AUC lift over no-skill  : {test.get('pr_auc_lift_over_no_skill', 0):.1f}x")
    if "leakage_demo" in results:
        print(f"Leakage demo PR-AUC        : {results['leakage_demo']['pr_auc']:.4f}  "
              f"<- NOT DEPLOYABLE, illustrates leakage inflation")
    if optimum:
        print(f"Cost-minimising threshold  : {optimum['optimal_threshold']:.2f} "
              f"(chosen on validation)")
        print(f"  fraud captured           : {100 * optimum['fraud_capture_rate']:.2f}%")
        print(f"  precision                : {100 * optimum['precision']:.2f}%")
        print(f"  legitimate blocked       : {optimum['legitimate_transactions_blocked']:,}")
        print(f"  simulated total cost     : ${optimum['total_cost_at_optimum']:,.0f}")
        print(f"  simulated cost at 0.50   : ${optimum['cost_at_threshold_0_50']:,.0f}")
    if sens:
        print(f"Sensitivity threshold range: {sens['threshold_min']:.2f} to "
              f"{sens['threshold_max']:.2f} across {sens['scenarios_tested']} scenarios")
    print("-" * 74)
    print("All monetary figures are SIMULATED under stated assumptions.")
    print(f"Reports  : {config.REPORT_DIR}")
    print(f"Figures  : {config.FIGURE_DIR}")
    print(f"Dashboard: {config.DASHBOARD_DIR}")
    print(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
