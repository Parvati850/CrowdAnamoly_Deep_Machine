"""
src/features/video_features.py
--------------------------------
Builds the feature matrix (X) and label vector (y) for the entire
labelled dataset by running motion_features.extract_video_features()
over every video, then saves the result as a compressed numpy archive.

This is the bridge between raw video data and model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.features.motion_features import extract_video_features, feature_names
from src.utils.logging_config import get_logger
from src.utils.paths import load_config, get_path, ensure_dir

logger = get_logger(__name__)

LABEL_MAP = {"normal": 0, "abnormal": 1}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def build_feature_matrix(
    manifest: pd.DataFrame,
    cfg: dict | None = None,
    split: str | None = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute motion feature vectors for all labelled, decodable videos.

    Parameters
    ----------
    manifest : DataFrame from inventory / split_dataset with a 'split' column.
    cfg      : Config dict.
    split    : If given, only process rows where manifest['split'] == split.
    overwrite: Re-extract even if a cached .npz exists.

    Returns
    -------
    (X, y, filenames)
      X         : float32 array of shape (N, n_features)
      y         : int array of shape (N,) with values 0=normal, 1=abnormal
      filenames : list of video filenames corresponding to each row
    """
    if cfg is None:
        cfg = load_config()

    interval = cfg["preprocessing"]["frame_sample_interval"]
    width    = cfg["preprocessing"]["target_width"]
    height   = cfg["preprocessing"]["target_height"]

    features_dir = get_path(cfg, "preprocessing", "features_output_dir")
    ensure_dir(features_dir)

    cache_tag = f"_{split}" if split else "_all"
    cache_path = features_dir / f"features{cache_tag}.npz"

    if cache_path.exists() and not overwrite:
        logger.info("Loading cached features from %s", cache_path)
        data = np.load(str(cache_path), allow_pickle=True)
        return (
            data["X"].astype(np.float32),
            data["y"].astype(int),
            list(data["filenames"]),
        )

    # Filter manifest
    subset = manifest[
        (manifest["source_group"] == "labelled") &
        (manifest["decoded"] == True)
    ].copy()

    if split is not None:
        subset = subset[subset["split"] == split]

    if subset.empty:
        logger.error("No videos to extract features from (split=%s).", split)
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int), []

    X_list, y_list, fnames = [], [], []
    failed = 0

    for _, row in subset.iterrows():
        video_path = Path(row["path"])
        label_str  = row["label"]

        if label_str not in LABEL_MAP:
            logger.warning("Skipping unknown label '%s' for %s", label_str, row["filename"])
            continue

        feat = extract_video_features(
            video_path,
            sample_interval=interval,
            target_width=width,
            target_height=height,
        )

        if feat is None:
            logger.warning("Feature extraction failed for %s", row["filename"])
            failed += 1
            continue

        X_list.append(feat)
        y_list.append(LABEL_MAP[label_str])
        fnames.append(row["filename"])

    if not X_list:
        logger.error("Feature extraction produced no vectors (split=%s).", split)
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int), []

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=int)

    np.savez_compressed(str(cache_path), X=X, y=y, filenames=np.array(fnames))
    logger.info(
        "Features saved: %s  shape=%s  failed=%d", cache_path.name, X.shape, failed
    )

    return X, y, fnames


def load_features(
    cfg: dict | None = None, split: str | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load pre-computed features from cache."""
    if cfg is None:
        cfg = load_config()
    features_dir = get_path(cfg, "preprocessing", "features_output_dir")
    cache_tag  = f"_{split}" if split else "_all"
    cache_path = features_dir / f"features{cache_tag}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {cache_path}\n"
            "Run scripts/prepare_dataset.py first."
        )
    data = np.load(str(cache_path), allow_pickle=True)
    return (
        data["X"].astype(np.float32),
        data["y"].astype(int),
        list(data["filenames"]),
    )
