"""Small shared helpers used by every stage."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


@contextmanager
def timed(label: str):
    """Log how long a stage took. Useful when a run takes 20 minutes."""
    log = logging.getLogger("timing")
    start = time.time()
    log.info("START  %s", label)
    yield
    log.info("DONE   %s (%.1fs)", label, time.time() - start)


def to_native(obj: Any) -> Any:
    """Make numpy/pandas objects JSON-serialisable."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, pd.Series):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_native(obj), indent=2), encoding="utf-8")
    return path


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion matrix plus the metrics that matter for imbalanced fraud.

    precision = TP / (TP + FP)   of everything flagged, how much was fraud
    recall    = TP / (TP + FN)   of all fraud, how much we caught
    fpr       = FP / (FP + TN)   share of legitimate traffic disrupted
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + tn + fp + fn
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "fraud_capture_rate": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "alert_volume": int(tp + fp),
        "alert_rate": float((tp + fp) / total) if total else 0.0,
    }


def fmt_money(value: float) -> str:
    return f"${value:,.0f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"
