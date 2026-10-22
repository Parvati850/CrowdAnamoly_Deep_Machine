"""
src/inference/video_inference.py
---------------------------------
Runs the trained model on a prerecorded video file and returns per-segment
anomaly scores and a final video-level prediction.

The video is processed in temporal windows:  every `window_size` sampled
frames, a feature vector is computed and a prediction is made.  This gives
a time-resolved anomaly score trace rather than a single video-level label.
"""

import time
import numpy as np
import cv2
from pathlib import Path
from typing import Generator

from src.features.motion_features import extract_video_features, feature_names
from src.models.baseline import load_model, predict as model_predict
from src.data.extract_frames import resize_frame
from src.inference.alerts import AlertManager
from src.utils.logging_config import get_logger
from src.utils.paths import load_config, get_path

logger = get_logger(__name__)

LABEL_NAMES = {0: "normal", 1: "abnormal"}


def analyse_video(
    video_path: Path,
    cfg: dict | None = None,
    model=None,
    alert_manager: AlertManager | None = None,
    window_size: int = 30,      # sampled frames per window
) -> dict:
    """
    Analyse a prerecorded video, returning a time-resolved score trace.

    Parameters
    ----------
    video_path    : Path to the video file.
    cfg           : Config dict.
    model         : Pre-loaded sklearn pipeline.
    alert_manager : AlertManager instance (optional).
    window_size   : Number of consecutive sampled frames per analysis window.

    Returns
    -------
    dict:
        windows          : list of dicts {start_frame, end_frame, score, label}
        overall_score    : mean abnormal score across all windows
        overall_label    : "normal" | "abnormal"
        fps_processing   : frames analysed per second
        alerts_fired     : number of alerts
        error            : None or error string
    """
    result = {
        "windows":        [],
        "overall_score":  None,
        "overall_label":  "unknown",
        "fps_processing": 0.0,
        "alerts_fired":   0,
        "error":          None,
    }

    if cfg is None:
        cfg = load_config()

    # Load model
    if model is None:
        ckpt = (get_path(cfg, "training", "model_output_dir")
                / cfg["training"]["checkpoint_name"])
        if not ckpt.exists():
            result["error"] = (
                f"No trained model found at {ckpt}. "
                "Run scripts/train_model.py first."
            )
            return result
        model = load_model(ckpt)

    interval = cfg["preprocessing"]["frame_sample_interval"]
    width    = cfg["preprocessing"]["target_width"]
    height   = cfg["preprocessing"]["target_height"]

    video_path = Path(video_path)
    if not video_path.exists():
        result["error"] = f"Video not found: {video_path}"
        return result

    # ── Stream frames in windows ────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["error"] = f"Cannot open video: {video_path.name}"
        return result

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev_gray   = None
    buffer      = []   # stores (frame_idx, gray_frame)
    frame_idx   = 0
    t_start     = time.perf_counter()
    frames_proc = 0

    windows = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                resized = resize_frame(frame, width, height)
                gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                buffer.append((frame_idx, gray))
                frames_proc += 1

                # When buffer is full, compute features and predict
                if len(buffer) >= window_size:
                    feat = _features_from_gray_buffer([g for _, g in buffer])
                    if feat is not None:
                        X = feat.reshape(1, -1)
                        try:
                            preds, probs = model_predict(model, X)
                            score = float(probs[0, 1])
                            label = LABEL_NAMES.get(int(preds[0]), "unknown")
                        except Exception as exc:
                            logger.warning("Prediction failed in window: %s", exc)
                            score, label = 0.0, "unknown"

                        win = {
                            "start_frame": buffer[0][0],
                            "end_frame":   buffer[-1][0],
                            "start_time_s": round(buffer[0][0] / fps_video, 2),
                            "end_time_s":   round(buffer[-1][0] / fps_video, 2),
                            "score":        round(score, 4),
                            "label":        label,
                        }
                        windows.append(win)

                        if alert_manager is not None:
                            # source_name is set on AlertManager at construction time
                            alert_manager.update(score, label)

                    buffer = []   # clear window

            frame_idx += 1

    finally:
        cap.release()  # guaranteed even if model_predict raises

    elapsed = time.perf_counter() - t_start

    result["windows"]        = windows
    result["fps_processing"] = round(frames_proc / elapsed, 1) if elapsed > 0 else 0.0
    result["alerts_fired"]   = alert_manager.active_count if alert_manager else 0

    if windows:
        scores = [w["score"] for w in windows]
        result["overall_score"] = round(float(np.mean(scores)), 4)
        result["overall_label"] = "abnormal" if result["overall_score"] >= cfg["alerts"]["threshold"] else "normal"

    return result


def _features_from_gray_buffer(grays: list) -> np.ndarray | None:
    """
    Compute motion features from a list of consecutive grayscale frames.
    Mirrors the logic in motion_features but operates on an in-memory buffer.
    """
    from src.features.motion_features import (
        _optical_flow_features, _frame_diff_features,
        N_DIR_BINS, HIGH_MOTION_THRESHOLD,
    )
    import cv2, numpy as np

    if len(grays) < 2:
        return None

    pair_stats = []
    for i in range(len(grays) - 1):
        f_of = _optical_flow_features(grays[i], grays[i + 1])
        f_fd = _frame_diff_features(grays[i], grays[i + 1])
        pair_stats.append({**f_of, **f_fd})

    keys   = list(pair_stats[0].keys())
    matrix = np.array([[ps[k] for k in keys] for ps in pair_stats])
    agg    = np.concatenate([matrix.mean(0), matrix.std(0), matrix.max(0), matrix.min(0)])
    extra  = np.array([float(len(pair_stats))])
    feat   = np.concatenate([agg, extra]).astype(np.float32)
    feat   = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat
