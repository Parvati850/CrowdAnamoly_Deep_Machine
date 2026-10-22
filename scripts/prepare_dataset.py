"""
scripts/prepare_dataset.py
---------------------------
Phase 3+4: Load manifest, assign splits, extract features.

Run:
    python scripts/prepare_dataset.py

Produces:
    dataset/dataset_manifest.csv   (updated with 'split' column)
    generated_data/features/features_train.npz
    generated_data/features/features_val.npz    (if enough data)
    generated_data/features/features_test.npz
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, PROJECT_ROOT
from src.utils.logging_config import setup_logging, get_logger
from src.data.inventory import load_inventory, save_inventory
from src.data.split_dataset import split_manifest, print_split_summary
from src.features.video_features import build_feature_matrix


def main():
    cfg = load_config()
    setup_logging(cfg["logging"]["level"],
                  PROJECT_ROOT / cfg["logging"]["log_file"])
    logger = get_logger("prepare_dataset")

    logger.info("=" * 60)
    logger.info("Phase 3+4: Dataset Preparation & Feature Extraction")
    logger.info("=" * 60)

    # ── Load manifest ───────────────────────────────────────────────────────
    manifest_path = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
    if not manifest_path.exists():
        logger.error(
            "Manifest not found at %s. Run scripts/inspect_dataset.py first.",
            manifest_path,
        )
        sys.exit(1)

    df = load_inventory(manifest_path)
    logger.info("Loaded manifest: %d rows", len(df))

    # ── Assign splits ───────────────────────────────────────────────────────
    df = split_manifest(
        df,
        test_size=cfg["training"]["test_size"],
        val_size=cfg["training"]["val_size"],
        random_seed=cfg["training"]["random_seed"],
    )
    print_split_summary(df)

    # Save manifest with split column
    save_inventory(df, manifest_path)
    logger.info("Manifest updated with split column.")

    # ── Extract features ────────────────────────────────────────────────────
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        if split_df.empty:
            logger.warning("No videos for split '%s' — skipping feature extraction.", split)
            continue

        logger.info("Extracting features for split: %s (%d videos)...", split, len(split_df))
        X, y, fnames = build_feature_matrix(df, cfg, split=split, overwrite=True)
        if len(X) > 0:
            logger.info(
                "  %s features: X=%s  classes=%s",
                split, X.shape,
                {0: "normal", 1: "abnormal"}
            )
            for i, fn in enumerate(fnames):
                logger.info("    [%d] %s → label=%d", i, fn, y[i])
        else:
            logger.warning("  Feature extraction for '%s' produced no vectors.", split)

    logger.info("Phase 3+4 complete.")


if __name__ == "__main__":
    main()
