"""
src/features/motion_features.py
---------------------------------
Extracts motion-based features from a video using dense optical flow
(Farneback method) and basic frame-difference statistics.

WHY THESE FEATURES?
  Abnormal crowd behaviour typically shows high-magnitude, disorganised,
  rapidly-changing motion.  The features below capture:
    - Mean and standard deviation of flow magnitude  → overall crowd speed
    - Flow uniformity (entropy of direction histogram) → how disorganised motion is
    - Fraction of high-motion pixels → proportion of crowd moving fast
    - Frame difference mean/std  → simpler pixel-change measure

All features are computed per PAIR of consecutive sampled frames, then
aggregated across the video into a fixed-length vector.  This means the
feature vector length is constant regardless of video duration.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Farneback optical flow parameters (tuned for 224×224 frames)
_OF_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# Pixels with flow magnitude above this threshold are "moving fast"
HIGH_MOTION_THRESHOLD = 2.0

# Number of bins for the direction histogram
N_DIR_BINS = 8


def _frame_to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to uint8 grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _optical_flow_features(prev_gray: np.ndarray, curr_gray: np.ndarray) -> dict:
    """
    Compute dense optical flow between two grayscale frames and return
    a dict of scalar statistics.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **_OF_PARAMS)

    # Magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mag_mean = float(np.mean(mag))
    mag_std  = float(np.std(mag))
    mag_max  = float(np.max(mag))

    # Fraction of pixels with high motion
    high_motion_frac = float(np.mean(mag > HIGH_MOTION_THRESHOLD))

    # Direction histogram entropy (low = uniform flow, high = chaotic)
    hist, _ = np.histogram(ang.ravel(), bins=N_DIR_BINS, range=(0, 2 * np.pi))
    hist    = hist.astype(float)
    hist   += 1e-10   # avoid log(0)
    hist   /= hist.sum()
    direction_entropy = float(-np.sum(hist * np.log2(hist)))  # max = log2(N_DIR_BINS)

    return {
        "of_mag_mean":       mag_mean,
        "of_mag_std":        mag_std,
        "of_mag_max":        mag_max,
        "of_high_motion_frac": high_motion_frac,
        "of_direction_entropy": direction_entropy,
    }


def _frame_diff_features(prev_gray: np.ndarray, curr_gray: np.ndarray) -> dict:
    """
    Compute frame-difference statistics (simpler than optical flow, faster).
    """
    diff  = cv2.absdiff(prev_gray, curr_gray).astype(np.float32)
    mean_ = float(np.mean(diff))
    std_  = float(np.std(diff))
    frac_ = float(np.mean(diff > 10))   # fraction of changed pixels

    return {
        "fd_mean":         mean_,
        "fd_std":          std_,
        "fd_changed_frac": frac_,
    }


def extract_video_features(
    video_path: Path,
    sample_interval: int = 10,
    target_width: int = 224,
    target_height: int = 224,
) -> Optional[np.ndarray]:
    """
    Compute a fixed-length feature vector for one video by:
      1. Sampling frames at sample_interval.
      2. Computing per-pair motion statistics.
      3. Aggregating with mean, std, max, min across all frame pairs.

    Returns None if the video cannot be decoded or yields < 2 sampled frames.

    Feature vector layout (33 values total):
      For each of 8 per-pair statistics: mean, std, max, min across pairs
      → 8 × 4 = 32 values
      Plus pair count: 1 value
      TOTAL = 33
    """
    from src.data.extract_frames import resize_frame   # local import to avoid circular

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open: %s", video_path.name)
        return None

    frames_gray = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            frame_resized = resize_frame(frame, target_width, target_height)
            frames_gray.append(_frame_to_gray(frame_resized))
        idx += 1
    cap.release()

    if len(frames_gray) < 2:
        logger.warning("Too few frames to compute motion features: %s", video_path.name)
        return None

    # Compute per-pair statistics
    pair_stats = []
    for i in range(len(frames_gray) - 1):
        f_of  = _optical_flow_features(frames_gray[i], frames_gray[i + 1])
        f_fd  = _frame_diff_features(frames_gray[i], frames_gray[i + 1])
        merged = {**f_of, **f_fd}
        pair_stats.append(merged)

    # Aggregate across pairs
    keys = list(pair_stats[0].keys())   # consistent order
    matrix = np.array([[ps[k] for k in keys] for ps in pair_stats])  # (N, 8)

    agg = np.concatenate([
        matrix.mean(axis=0),
        matrix.std(axis=0),
        matrix.max(axis=0),
        matrix.min(axis=0),
    ])  # 8 × 4 = 32 values

    extra = np.array([float(len(pair_stats))])   # number of frame pairs
    feature_vector = np.concatenate([agg, extra])  # 33 values total

    if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
        logger.warning("NaN/Inf in feature vector for %s – replacing with zeros.", video_path.name)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector.astype(np.float32)


def feature_names() -> list[str]:
    """Return human-readable names matching the extract_video_features output."""
    base_keys = [
        "of_mag_mean", "of_mag_std", "of_mag_max",
        "of_high_motion_frac", "of_direction_entropy",
        "fd_mean", "fd_std", "fd_changed_frac",
    ]
    agg_ops = ["mean", "std", "max", "min"]
    names = [f"{k}_{op}" for op in agg_ops for k in base_keys]
    names.append("n_frame_pairs")
    return names
