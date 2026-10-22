"""
scripts/train_model.py
-----------------------
Phase 5: Train the baseline model.

Run:
    python scripts/train_model.py

Produces:
    models/baseline_model.pkl
    outputs/reports/train_metadata.json
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, PROJECT_ROOT, get_path
from src.utils.logging_config import setup_logging, get_logger
from src.data.inventory import load_inventory
from src.models.train import run_training
from src.models.baseline import predict as model_predict


def main():
    cfg = load_config()
    setup_logging(cfg["logging"]["level"],
                  PROJECT_ROOT / cfg["logging"]["log_file"])
    logger = get_logger("train_model")

    logger.info("=" * 60)
    logger.info("Phase 5: Model Training")
    logger.info("=" * 60)

    # ── Load manifest ───────────────────────────────────────────────────────
    manifest_path = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
    if not manifest_path.exists():
        logger.error(
            "Manifest not found. Run prepare_dataset.py first."
        )
        sys.exit(1)

    df = load_inventory(manifest_path)

    if "split" not in df.columns:
        logger.error(
            "'split' column missing from manifest. Run prepare_dataset.py first."
        )
        sys.exit(1)

    train_count = (df["split"] == "train").sum()
    if train_count == 0:
        logger.error("No training videos found in manifest.")
        sys.exit(1)

    # ── Run training ────────────────────────────────────────────────────────
    result = run_training(df, cfg, overwrite_features=False)

    logger.info("Training complete. Checkpoint: %s", result["checkpoint_path"])
    logger.info(
        "Train set: %d videos | Val set: %d videos",
        len(result["train_filenames"]),
        len(result["val_filenames"]),
    )

    # Quick sanity check: run predict on the training set itself
    model  = result["model"]
    X_tr   = result["X_train"]
    y_tr   = result["y_train"]

    preds, probs = model_predict(model, X_tr)
    train_acc = (preds == y_tr).mean()
    logger.info(
        "Training-set accuracy (not evaluation metric): %.3f  "
        "(expected to be high — evaluate on held-out test set)",
        train_acc,
    )

    # Save training metadata
    reports_dir = get_path(cfg, "output", "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "checkpoint":       str(result["checkpoint_path"]),
        "classifier_type":  cfg["model"]["baseline_classifier"],
        "n_train":          int(len(result["train_filenames"])),
        "n_val":            int(len(result["val_filenames"])),
        "train_filenames":  result["train_filenames"],
        "val_filenames":    result["val_filenames"],
        "train_accuracy_on_trainset": round(float(train_acc), 4),
    }
    meta_path = reports_dir / "train_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Training metadata saved to %s", meta_path)

    print(f"\n  Model saved to: {result['checkpoint_path']}")
    print(f"  Train videos:   {len(result['train_filenames'])}")
    print(f"  Val   videos:   {len(result['val_filenames'])}")
    print(f"  Training-set accuracy: {train_acc:.3f}")
    print("\n  Next step: run scripts/evaluate_model.py\n")


if __name__ == "__main__":
    main()
