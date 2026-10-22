"""
scripts/evaluate_deep_model.py
--------------------------------
Evaluate the trained CNN-LSTM on the held-out test split.
Uses the SAME 6 test videos as the Random Forest evaluation.
Results saved alongside the RF results for dashboard comparison.

Run:
    .\.venv\Scripts\python.exe scripts/evaluate_deep_model.py
"""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.utils.paths import load_config, get_path, PROJECT_ROOT
from src.utils.logging_config import setup_logging, get_logger
from src.data.inventory import load_inventory
from src.models.deep_predict import predict_video_deep
from src.models.deep_model import load_deep_model
from src.models.evaluate import compute_metrics, print_metrics, plot_confusion_matrix, save_metrics

cfg = load_config()
setup_logging(cfg["logging"]["level"])
logger = get_logger("evaluate_deep_model")


def main():
    logger.info("=" * 60)
    logger.info("CNN-LSTM Evaluation")
    logger.info("=" * 60)

    ckpt_path = get_path(cfg, "training", "model_output_dir") / "deep_model.pth"
    if not ckpt_path.exists():
        logger.error("No checkpoint found. Run train_deep_model.py first.")
        sys.exit(1)

    model = load_deep_model(ckpt_path)
    model.eval()
    logger.info("Loaded checkpoint: %s", ckpt_path)

    manifest_path = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
    df = load_inventory(manifest_path)
    test_df = df[(df["source_group"] == "labelled") &
                 (df["decoded"] == True) &
                 (df["split"] == "test")]

    if test_df.empty:
        logger.error("No test videos. Run prepare_dataset.py first.")
        sys.exit(1)

    logger.info("Test set: %d videos", len(test_df))
    LABEL_MAP = {"normal": 0, "abnormal": 1}
    LABEL_NAMES = {0: "normal", 1: "abnormal"}

    y_true, y_pred, y_proba, fnames = [], [], [], []
    for _, row in test_df.iterrows():
        logger.info("  Predicting: %s", row["filename"])
        r = predict_video_deep(Path(row["path"]), cfg=cfg, model=model)
        if r["error"]:
            logger.warning("    FAILED: %s", r["error"])
            continue
        true_id = LABEL_MAP[row["label"]]
        y_true.append(true_id)
        y_pred.append(r["label_id"])
        y_proba.append([r["score_normal"], r["score_abnormal"]])
        fnames.append(row["filename"])
        logger.info("    true=%s  pred=%s  score=%.4f",
                    row["label"], r["label"], r["score_abnormal"])

    if not y_true:
        logger.error("No predictions. Evaluation aborted.")
        sys.exit(1)

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_proba = np.array(y_proba)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["test_filenames"]   = fnames
    metrics["test_true_labels"] = y_true.tolist()
    metrics["test_predictions"] = y_pred.tolist()
    metrics["model_type"]       = "cnn_lstm"

    print_metrics(metrics)

    reports_dir = get_path(cfg, "output", "reports_dir")
    plots_dir   = get_path(cfg, "output", "plots_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics, reports_dir / "evaluation_metrics_deep.json")
    plot_confusion_matrix(y_true, y_pred, plots_dir / "confusion_matrix_deep.png")

    print(f"\n  Deep model metrics saved to: {reports_dir / 'evaluation_metrics_deep.json'}")
    print(f"  Next: run app.py and select CNN-LSTM model")


if __name__ == "__main__":
    main()
