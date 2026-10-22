"""
src/models/train.py
--------------------
Orchestrates the full training workflow:
  1. Load manifest
  2. Build features for train split
  3. (Optionally build val split features)
  4. Train baseline model
  5. Save checkpoint
  6. Return training metadata
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.models.baseline import train as train_baseline, save_model
from src.features.video_features import build_feature_matrix
from src.utils.logging_config import get_logger
from src.utils.paths import load_config, get_path

logger = get_logger(__name__)


def run_training(
    manifest: pd.DataFrame,
    cfg: dict | None = None,
    overwrite_features: bool = False,
) -> dict:
    """
    End-to-end training run.

    Parameters
    ----------
    manifest           : DataFrame with 'split' column (from split_dataset).
    cfg                : Config dict.
    overwrite_features : Re-extract features even if cache exists.

    Returns
    -------
    dict with keys: model, checkpoint_path, X_train, y_train, X_val, y_val,
                    train_filenames, val_filenames, cfg
    """
    if cfg is None:
        cfg = load_config()

    # ── Build feature matrices ───────────────────────────────────────────────
    logger.info("Building training features...")
    X_train, y_train, train_fnames = build_feature_matrix(
        manifest, cfg, split="train", overwrite=overwrite_features
    )

    if len(X_train) == 0:
        raise RuntimeError(
            "Training feature matrix is empty. "
            "Ensure there are labelled decodable videos in the 'train' split."
        )

    logger.info("Training set: %d videos, %d features", *X_train.shape)
    logger.info(
        "  Label distribution: normal=%d, abnormal=%d",
        (y_train == 0).sum(), (y_train == 1).sum()
    )

    # Val set (optional — may be empty if dataset too small)
    X_val, y_val, val_fnames = build_feature_matrix(
        manifest, cfg, split="val", overwrite=overwrite_features
    )
    if len(X_val) > 0:
        logger.info("Validation set: %d videos", len(X_val))
    else:
        logger.warning("No validation split available.")

    # ── Train ───────────────────────────────────────────────────────────────
    model = train_baseline(X_train, y_train, cfg)

    # ── Save ────────────────────────────────────────────────────────────────
    models_dir     = get_path(cfg, "training", "model_output_dir")
    checkpoint_name = cfg["training"]["checkpoint_name"]
    checkpoint_path = models_dir / checkpoint_name

    save_model(model, checkpoint_path)

    return {
        "model":           model,
        "checkpoint_path": checkpoint_path,
        "X_train":         X_train,
        "y_train":         y_train,
        "X_val":           X_val,
        "y_val":           y_val,
        "train_filenames": train_fnames,
        "val_filenames":   val_fnames,
        "cfg":             cfg,
    }
