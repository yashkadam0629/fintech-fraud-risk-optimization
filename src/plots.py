"""
Figure generation for every stage.

Rule for this module: each figure answers one stated question, and the
question is written into the title so the image stands alone in a README or a
slide. Every function returns the path it wrote and never raises upward.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

FRAUD = "#c0392b"
LEGIT = "#2c7fb8"
NEUTRAL = "#7f8c8d"
ACCENT = "#e67e22"


def _save(fig, name: str, directory: Path | None = None) -> Path:
    directory = Path(directory) if directory else config.FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  figure: %s", name)
    return path


def safe(func):
    """Decorator: a failed chart logs and returns None, it does not kill a run."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Figure %s failed", func.__name__)
            return None
    wrapper.__name__ = func.__name__
    return wrapper


# ==========================================================================
# Stage 1
# ==========================================================================
@safe
def plot_class_imbalance(df: pd.DataFrame) -> Path:
    counts = df[config.TARGET].value_counts().reindex([0, 1]).fillna(0)
    rate = 100 * counts.get(1, 0) / counts.sum()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Legitimate", "Fraudulent"], [counts.get(0, 0), counts.get(1, 0)],
                  color=[LEGIT, FRAUD])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(f"How rare is fraud?  Fraud rate = {rate:.4f}%")
    for bar, value in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}",
                ha="center", va="bottom", fontsize=10)
    return _save(fig, "01_class_imbalance.png")


@safe
def plot_amount_distribution(df: pd.DataFrame) -> Path:
    positive = df.loc[df["amount"] > 0]
    log_amount = np.log10(positive["amount"].to_numpy())
    is_fraud = positive[config.TARGET].to_numpy() == 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(log_amount, bins=60, color=NEUTRAL)
    axes[0].set_xlabel("log10(amount)")
    axes[0].set_ylabel("Transactions")
    axes[0].set_title("Amount distribution\n(log scale: amounts are heavily skewed)")

    bins = np.linspace(log_amount.min(), log_amount.max(), 60)
    axes[1].hist(log_amount[~is_fraud], bins=bins, density=True, alpha=0.6,
                 color=LEGIT, label="Legitimate")
    if is_fraud.any():
        axes[1].hist(log_amount[is_fraud], bins=bins, density=True, alpha=0.6,
                     color=FRAUD, label="Fraudulent")
        axes[1].axvline(np.median(log_amount[is_fraud]), color=FRAUD, ls="--")
    axes[1].axvline(np.median(log_amount[~is_fraud]), color=LEGIT, ls="--")
    axes[1].set_xlabel("log10(amount)")
    axes[1].set_ylabel("Density (each class sums to 1)")
    axes[1].set_title("Do fraudulent amounts differ?\nDashed lines = medians")
    axes[1].legend()
    return _save(fig, "02_amount_distribution.png")


@safe
def plot_fraud_by_type(df: pd.DataFrame) -> Path:
    grouped = (df.groupby("type", observed=True)
               .agg(transactions=("amount", "size"), fraud=(config.TARGET, "sum")))
    grouped["fraud_rate"] = 100 * grouped["fraud"] / grouped["transactions"]
    grouped = grouped.sort_values("transactions", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(grouped.index.astype(str), grouped["transactions"], color=LEGIT)
    axes[0].set_ylabel("Transactions")
    axes[0].set_title("Volume by transaction type")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(grouped.index.astype(str), grouped["fraud_rate"], color=FRAUD)
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].set_title("Where does fraud actually occur?\nBar labels = fraud counts")
    axes[1].tick_params(axis="x", rotation=30)
    for i, (rate, count) in enumerate(zip(grouped["fraud_rate"], grouped["fraud"])):
        axes[1].text(i, rate, f"{int(count):,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, "03_fraud_by_type.png")


@safe
def plot_hourly_pattern(df: pd.DataFrame) -> Path:
    hourly = (df.groupby("hour_of_day", observed=True)
              .agg(transactions=("amount", "size"), fraud=(config.TARGET, "sum"))
              .reindex(range(24)).fillna(0))
    hourly["fraud_rate"] = 100 * hourly["fraud"] / hourly["transactions"].replace(0, np.nan)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(hourly.index, hourly["transactions"], color=LEGIT, alpha=0.75)
    ax1.set_xlabel("Hour within the 24-hour cycle (real clock time unknown)")
    ax1.set_ylabel("Transactions", color=LEGIT)
    ax1.set_xticks(range(24))
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["fraud_rate"], color=FRAUD, marker="o", lw=2)
    ax2.set_ylabel("Fraud rate (%)", color=FRAUD)
    ax1.set_title("Does fraud follow normal activity hours?\n"
                  "A fraud-rate spike where volume collapses is a simulator "
                  "artifact, not customer behaviour")
    return _save(fig, "04_hourly_pattern.png")


@safe
def plot_account_repetition(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, label in ((axes[0], "orig_id", "Origin accounts"),
                           (axes[1], "dest_id", "Destination accounts")):
        _, counts = np.unique(df[col].to_numpy(), return_counts=True)
        ax.hist(np.clip(counts, 1, 20), bins=np.arange(0.5, 21.5, 1), color=NEUTRAL)
        ax.set_yscale("log")
        ax.set_xlabel("Transactions per account (clipped at 20)")
        ax.set_ylabel("Accounts (log scale)")
        once = 100 * (counts == 1).sum() / counts.size
        ax.set_title(f"{label}\n{once:.1f}% appear once ({counts.size:,} accounts)")
    fig.suptitle("Can per-customer behavioural baselines be computed?", y=1.03)
    return _save(fig, "05_account_repetition.png")


@safe
def plot_leakage_separation(audit: dict) -> Path:
    checks = audit["balance_pattern_checks_pct"]
    names = list(checks.keys())
    fraud_values = [checks[n]["fraud_pct"] for n in names]
    legit_values = [checks[n]["legitimate_pct"] for n in names]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(y - 0.2, fraud_values, height=0.4, color=FRAUD, label="Fraudulent")
    ax.barh(y + 0.2, legit_values, height=0.4, color=LEGIT, label="Legitimate")
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    ax.set_xlabel("Share of transactions (%)")
    ax.set_title("Leakage audit: do balance fields encode the outcome?\n"
                 "Wide gaps mean the column reveals the label")
    ax.legend()
    return _save(fig, "06_leakage_audit.png")


# ==========================================================================
# Stage 3
# ==========================================================================
@safe
def plot_model_curves(curves: dict[str, dict], prevalence: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, curve in curves.items():
        axes[0].plot(curve["fpr"], curve["tpr"], lw=2, label=name)
        axes[1].plot(curve["recall"], curve["precision"], lw=2, label=name)
    axes[0].plot([0, 1], [0, 1], ls="--", color=NEUTRAL, label="random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve\nFlattering under heavy imbalance")
    axes[0].legend(fontsize=8)

    axes[1].axhline(prevalence, ls="--", color=NEUTRAL,
                    label=f"no skill ({prevalence:.4f})")
    axes[1].set_xlabel("Recall (fraud captured)")
    axes[1].set_ylabel("Precision (alert quality)")
    axes[1].set_title("Precision-Recall curve\nThe metric that matters here")
    axes[1].legend(fontsize=8)
    return _save(fig, "07_model_curves.png")


# ==========================================================================
# Stage 3B / 3C
# ==========================================================================
@safe
def plot_threshold_cost(sweep: pd.DataFrame, optimum: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(sweep["threshold"], sweep["total_cost"], color="black", lw=2,
                 label="Total simulated cost")
    axes[0].plot(sweep["threshold"], sweep["cost_missed_fraud"], color=FRAUD, ls="--",
                 label="Missed-fraud cost")
    axes[0].plot(sweep["threshold"], sweep["cost_customer_friction"], color=LEGIT, ls="--",
                 label="Customer-friction cost")
    axes[0].axvline(optimum["optimal_threshold"], color=ACCENT, lw=2)
    axes[0].annotate(f"min cost at t = {optimum['optimal_threshold']:.2f}",
                     xy=(optimum["optimal_threshold"], optimum["total_cost_at_optimum"]),
                     xytext=(0.45, 0.8), textcoords="axes fraction", color=ACCENT)
    axes[0].axvline(0.5, color=NEUTRAL, ls=":", label="conventional 0.50")
    axes[0].set_xlabel("Intervention threshold")
    axes[0].set_ylabel("Simulated cost")
    # Dollar signs are escaped: an unescaped pair puts matplotlib into mathtext
    # mode and silently italicises everything between them.
    axes[0].set_title(f"Where does total cost bottom out?\n"
                      f"SIMULATED: FN=\\${config.COST_FALSE_NEGATIVE:.0f}, "
                      f"FP=\\${config.COST_FALSE_POSITIVE:.0f} per event")
    axes[0].legend(fontsize=8)

    axes[1].plot(sweep["threshold"], 100 * sweep["fraud_capture_rate"], color=FRAUD,
                 lw=2, label="Fraud captured (%)")
    ax2 = axes[1].twinx()
    ax2.plot(sweep["threshold"], sweep["legitimate_transactions_blocked"], color=LEGIT,
             lw=2, label="Legitimate blocked")
    axes[1].axvline(optimum["optimal_threshold"], color=ACCENT, lw=2)
    axes[1].set_xlabel("Intervention threshold")
    axes[1].set_ylabel("Fraud captured (%)", color=FRAUD)
    ax2.set_ylabel("Legitimate transactions blocked", color=LEGIT)
    axes[1].set_title("The trade-off being priced\nFraud caught vs customers disrupted")
    return _save(fig, "08_threshold_vs_cost.png")


@safe
def plot_sensitivity_heatmaps(grid: pd.DataFrame) -> Path:
    from src.sensitivity_analysis import threshold_matrix

    threshold_pivot = threshold_matrix(grid, "optimal_threshold")
    capture_pivot = threshold_matrix(grid, "fraud_capture_rate")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pivot, title, fmt, cmap in (
        (axes[0], threshold_pivot, "Minimum-cost threshold", "{:.2f}", "viridis"),
        (axes[1], capture_pivot * 100, "Fraud captured at that threshold (%)", "{:.1f}", "magma"),
    ):
        image = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"${c:,.0f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"${i:,.0f}" for i in pivot.index])
        ax.set_xlabel("Cost of one false positive (friction)")
        ax.set_ylabel("Cost of one missed fraud")
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, fmt.format(pivot.values[i, j]), ha="center",
                        va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("The operating threshold is a function of business economics "
                 "(SIMULATED assumptions)", y=1.02)
    return _save(fig, "09_cost_sensitivity_heatmap.png")


# ==========================================================================
# Stage 3D / 4
# ==========================================================================
@safe
def plot_feature_importance(importance: pd.DataFrame, top_n: int = 20,
                            name: str = "10_feature_importance.png") -> Path:
    top = importance.head(top_n).iloc[::-1]
    value_column = top.columns[1]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(top) + 2))
    ax.barh(top["feature"], top[value_column], color=NEUTRAL)
    ax.set_xlabel(value_column.replace("_", " "))
    ax.set_title("Which features contributed most to the model's scores?\n"
                 "Association with the prediction, not proof of causation")
    return _save(fig, name)


@safe
def plot_false_positive_profile(scored: pd.DataFrame) -> Path:
    legitimate = scored[scored[config.TARGET] == 0]
    by_type = (legitimate.assign(fp=(legitimate["outcome_class"] == "false_positive"))
               .groupby("type", observed=True)["fp"].agg(["sum", "mean", "size"]))
    by_band = (legitimate.assign(fp=(legitimate["outcome_class"] == "false_positive"))
               .groupby("amount_band", observed=True)["fp"].agg(["sum", "mean", "size"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(by_type.index.astype(str), 100 * by_type["mean"], color=ACCENT)
    axes[0].set_ylabel("False positive rate (%)")
    axes[0].set_title("Where is customer friction concentrated?\nBy transaction type")
    axes[0].tick_params(axis="x", rotation=30)
    for i, count in enumerate(by_type["sum"]):
        axes[0].text(i, 100 * by_type["mean"].iloc[i], f"{int(count):,}",
                     ha="center", va="bottom", fontsize=9)

    axes[1].bar(by_band.index.astype(str), 100 * by_band["mean"], color=ACCENT)
    axes[1].set_ylabel("False positive rate (%)")
    axes[1].set_title("By transaction amount band")
    axes[1].tick_params(axis="x", rotation=30)
    return _save(fig, "11_false_positive_profile.png")


@safe
def plot_risk_band_distribution(scored: pd.DataFrame) -> Path:
    order = ["LOW", "MEDIUM", "HIGH"]
    counts = scored["risk_band"].value_counts().reindex(order).fillna(0)
    fraud_counts = (scored[scored[config.TARGET] == 1]["risk_band"]
                    .value_counts().reindex(order).fillna(0))
    rate = 100 * fraud_counts / counts.replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(order, counts.values, color=[LEGIT, ACCENT, FRAUD])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Transactions (log scale)")
    axes[0].set_title("How much volume lands in each derived risk band?")
    axes[1].bar(order, rate.values, color=[LEGIT, ACCENT, FRAUD])
    axes[1].set_ylabel("Fraud rate within band (%)")
    axes[1].set_title("Do the bands separate risk?")
    return _save(fig, "12_risk_bands.png")
