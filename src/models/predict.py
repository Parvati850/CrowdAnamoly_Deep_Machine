"""
src/models/predict.py
----------------------
Single-video inference using the trained baseline model.
Returns an anomaly score and classification for a given video file.
"""

import numpy as np
from pathlib import Path

from src.models.baseline import load_model, predict as model_predict
from src.features.motion_features import extract_video_features
from src.utils.logging_config import get_logger
from src.utils.paths import load_config, get_path

logger = get_logger(__name__)

LABEL_NAMES = {0: "normal", 1: "abnormal"}


def predict_video(
    video_path: Path,
    cfg: dict | None = None,
    model=None,
) -> dict:
    """
    Run inference on a single video file.

    Parameters
    ----------
    video_path : Path to the video.
    cfg        : Config dict.
    model      : Pre-loaded pipeline (avoids reloading on every call).

    Returns
    -------
    dict:
        label         : "normal" | "abnormal" | "error"
        label_id      : 0 | 1 | -1
        score_abnormal: float in [0,1], probability of being abnormal
        score_normal  : float in [0,1], probability of being normal
        model_loaded  : bool
        feature_ok    : bool
        error         : str | None
    """
    result = {
        "label":          "error",
        "label_id":       -1,
        "score_abnormal": 0.0,
        "score_normal":   0.0,
        "model_loaded":   False,
        "feature_ok":     False,
        "error":          None,
    }

    if cfg is None:
        cfg = load_config()

    # ── Load model ────────────────────────────────────────────────────────
    if model is None:
        checkpoint_path = (
            get_path(cfg, "training", "model_output_dir")
            / cfg["training"]["checkpoint_name"]
        )
        if not checkpoint_path.exists():
            result["error"] = (
                f"Model checkpoint not found at {checkpoint_path}. "
                "Please run scripts/train_model.py first."
            )
            logger.error(result["error"])
            return result

        try:
            model = load_model(checkpoint_path)
            result["model_loaded"] = True
        except Exception as exc:
            result["error"] = f"Failed to load model: {exc}"
            logger.error(result["error"])
            return result
    else:
        result["model_loaded"] = True

    # ── Extract features ─────────────────────────────────────────────────
    video_path = Path(video_path)
    if not video_path.exists():
        result["error"] = f"Video file not found: {video_path}"
        return result

    interval = cfg["preprocessing"]["frame_sample_interval"]
    width    = cfg["preprocessing"]["target_width"]
    height   = cfg["preprocessing"]["target_height"]

    feat = extract_video_features(video_path, interval, width, height)
    if feat is None:
        result["error"] = "Feature extraction failed (video may be unreadable or too short)."
        return result

    result["feature_ok"] = True

    # ── Predict ────────────────────────────────────────────────────────────
    try:
        X = feat.reshape(1, -1)
        labels, probs = model_predict(model, X)
        label_id = int(labels[0])
        result["label"]          = LABEL_NAMES.get(label_id, "unknown")
        result["label_id"]       = label_id
        result["score_normal"]   = round(float(probs[0, 0]), 4)
        result["score_abnormal"] = round(float(probs[0, 1]), 4)
    except Exception as exc:
        result["error"] = f"Prediction failed: {exc}"
        logger.error(result["error"])

    return result
