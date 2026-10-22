"""
src/models/evaluate.py
-----------------------
Computes evaluation metrics on a held-out test set.

Metrics reported:
  - Confusion matrix
  - Precision, Recall, F1 per class
  - Balanced accuracy
  - ROC-AUC (when applicable)
  - False-positive and false-negative counts

IMPORTANT: Metrics are computed only from actual model predictions on
held-out data.  No metric is fabricated or assumed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, classification_report,
    balanced_accuracy_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

from src.utils.logging_config import get_logger
from src.utils.paths import get_path, ensure_dir

logger = get_logger(__name__)

LABEL_NAMES = ["normal", "abnormal"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true  : Ground-truth integer labels.
    y_pred  : Predicted integer labels.
    y_proba : Predicted probabilities, shape (N, 2).  Used for ROC-AUC.

    Returns
    -------
    dict with metric names and values / sub-dicts.
    """
    if len(y_true) == 0:
        return {"error": "Empty test set – no metrics computed."}

    present_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    # Only pass target_names when both classes are present to avoid sklearn crash
    tn_names = [LABEL_NAMES[i] for i in present_classes] if len(present_classes) == len(LABEL_NAMES) else None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # False positive / negative counts
    # FP = predicted abnormal but actually normal
    # FN = predicted normal but actually abnormal
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "balanced_accuracy": round(bal_acc, 4),
        "per_class": report,
        "roc_auc": None,
    }

    # ROC-AUC requires probability scores and both classes present
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1])
            metrics["roc_auc"] = round(float(auc), 4)
        except Exception as exc:
            logger.warning("ROC-AUC could not be computed: %s", exc)

    return metrics


def print_metrics(metrics: dict) -> None:
    """Human-readable metrics summary."""
    if "error" in metrics:
        print(f"[EVALUATION ERROR] {metrics['error']}")
        return

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS (held-out test set)")
    print("=" * 60)
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")

    print("\n  Per-class metrics:")
    for label in LABEL_NAMES:
        r = metrics["per_class"].get(label, {})
        print(
            f"    {label:<10} precision={r.get('precision',0):.3f}  "
            f"recall={r.get('recall',0):.3f}  "
            f"f1={r.get('f1-score',0):.3f}  "
            f"support={r.get('support',0)}"
        )

    print("\n  Confusion matrix (rows=actual, cols=predicted):")
    cm = np.array(metrics["confusion_matrix"])
    print(f"              pred_normal  pred_abnormal")
    for i, row_label in enumerate(LABEL_NAMES):
        print(f"  actual_{row_label:<10} {cm[i, 0]:>11}  {cm[i, 1]:>13}")
    print("=" * 60 + "\n")

    # Practical interpretation
    print("  Practical interpretation:")
    if metrics["fn"] > 0:
        print(f"  ⚠ {metrics['fn']} abnormal video(s) missed (false negatives).")
    if metrics["fp"] > 0:
        print(f"  ⚠ {metrics['fp']} normal video(s) flagged as abnormal (false positives).")
    if metrics["fn"] == 0 and metrics["fp"] == 0:
        print("  ✓ Perfect on test set (note: test set is very small).")
    print()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Save a labelled confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix – Test Set", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Confusion matrix saved to %s", output_path)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Save a ROC curve plot (only when both classes are present)."""
    if len(np.unique(y_true)) < 2:
        logger.warning("ROC curve skipped: only one class in test set.")
        return

    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(
            y_true, y_proba[:, 1],
            name="Baseline model", ax=ax
        )
        ax.set_title("ROC Curve – Test Set", fontsize=12, fontweight="bold")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        plt.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            logger.info("ROC curve saved to %s", output_path)
        plt.close(fig)
    except Exception as exc:
        logger.warning("ROC curve plot failed: %s", exc)


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics dict to a JSON file."""
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Metrics saved to %s", output_path)
