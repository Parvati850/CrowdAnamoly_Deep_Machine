"""
scripts/evaluate_model.py
--------------------------
Phase 6: Evaluate the trained model on the held-out test set.

Run:
    python scripts/evaluate_model.py

Produces:
    outputs/reports/evaluation_metrics.json
    outputs/plots/confusion_matrix.png
    outputs/plots/roc_curve.png
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, PROJECT_ROOT, get_path
from src.utils.logging_config import setup_logging, get_logger
from src.data.inventory import load_inventory
from src.models.baseline import load_model, predict as model_predict
from src.features.video_features import build_feature_matrix, load_features
from src.models.evaluate import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curve, save_metrics,
)
import numpy as np


def main():
    cfg = load_config()
    setup_logging(cfg["logging"]["level"],
                  PROJECT_ROOT / cfg["logging"]["log_file"])
    logger = get_logger("evaluate_model")

    logger.info("=" * 60)
    logger.info("Phase 6: Model Evaluation")
    logger.info("=" * 60)

    # ── Load model ──────────────────────────────────────────────────────────
    models_dir = get_path(cfg, "training", "model_output_dir")
    ckpt_path  = models_dir / cfg["training"]["checkpoint_name"]
    if not ckpt_path.exists():
        logger.error("Model checkpoint not found at %s. Run train_model.py first.", ckpt_path)
        sys.exit(1)

    model = load_model(ckpt_path)

    # ── Load test features ──────────────────────────────────────────────────
    manifest_path = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
    df = load_inventory(manifest_path)

    if "split" not in df.columns:
        logger.error("'split' column missing. Run prepare_dataset.py first.")
        sys.exit(1)

    test_df = df[df["split"] == "test"]
    if test_df.empty:
        logger.warning(
            "No test split found. Evaluating on ALL labelled decodable videos "
            "(this gives an overly optimistic estimate due to train/test overlap)."
        )
        # Fall back to full labelled decodable set
        X, y, fnames = build_feature_matrix(df, cfg, split=None, overwrite=False)
    else:
        try:
            X, y, fnames = load_features(cfg, split="test")
        except FileNotFoundError:
            logger.info("Test feature cache not found — extracting now...")
            X, y, fnames = build_feature_matrix(df, cfg, split="test", overwrite=True)

    if len(X) == 0:
        logger.error("Test feature matrix is empty. Cannot evaluate.")
        sys.exit(1)

    logger.info("Test set: %d videos", len(X))
    for i, fn in enumerate(fnames):
        logger.info("  [%d] %s → true_label=%d", i, fn, y[i])

    # ── Predict ─────────────────────────────────────────────────────────────
    preds, probs = model_predict(model, X)
    logger.info("Predictions: %s", preds.tolist())
    logger.info("True labels: %s", y.tolist())

    # ── Metrics ─────────────────────────────────────────────────────────────
    metrics = compute_metrics(y, preds, probs)
    print_metrics(metrics)

    # ── Save outputs ─────────────────────────────────────────────────────────
    reports_dir = get_path(cfg, "output", "reports_dir")
    plots_dir   = get_path(cfg, "output", "plots_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "evaluation_metrics.json"

    # Add test filenames to the metrics output for traceability
    metrics["test_filenames"] = fnames
    metrics["test_true_labels"]  = y.tolist()
    metrics["test_predictions"]  = preds.tolist()

    save_metrics(metrics, metrics_path)

    plot_confusion_matrix(y, preds, plots_dir / "confusion_matrix.png")
    if probs is not None:
        plot_roc_curve(y, probs, plots_dir / "roc_curve.png")

    print(f"\n  Evaluation metrics saved to: {metrics_path}")
    print(f"  Confusion matrix plot:       {plots_dir / 'confusion_matrix.png'}")
    print(f"  ROC curve plot:              {plots_dir / 'roc_curve.png'}")
    print("\n  IMPORTANT: The test set contains very few videos.")
    print("  These metrics are indicative only. Do not over-interpret them.")
    print("\n  Next step: python app.py  (to launch the dashboard)\n")


if __name__ == "__main__":
    main()
