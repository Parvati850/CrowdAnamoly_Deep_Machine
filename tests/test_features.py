"""
tests/test_features.py
-----------------------
Unit tests for motion feature extraction.
Uses the synthetic_video fixture — no original files touched.
"""

import pytest
import numpy as np
from pathlib import Path


def test_feature_vector_shape(synthetic_video, project_root):
    """Feature extraction returns a 1D vector of expected length."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.features.motion_features import extract_video_features, feature_names

    feat = extract_video_features(synthetic_video, sample_interval=2, target_width=64, target_height=64)

    assert feat is not None, "Feature extraction returned None for a valid video"
    assert isinstance(feat, np.ndarray)
    assert feat.ndim == 1
    assert len(feat) == len(feature_names()), (
        f"Feature vector length {len(feat)} != expected {len(feature_names())}"
    )


def test_feature_vector_no_nan(synthetic_video, project_root):
    """Feature vector contains no NaN or Inf values."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.features.motion_features import extract_video_features

    feat = extract_video_features(synthetic_video, sample_interval=2, target_width=64, target_height=64)
    assert feat is not None
    assert not np.isnan(feat).any(), "NaN in feature vector"
    assert not np.isinf(feat).any(), "Inf in feature vector"


def test_feature_names_count(project_root):
    """feature_names() returns the correct number of names."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.features.motion_features import feature_names

    names = feature_names()
    # 8 base features × 4 aggregation ops + 1 count = 33
    assert len(names) == 33, f"Expected 33 feature names, got {len(names)}"


def test_invalid_video_returns_none(tmp_path, project_root):
    """extract_video_features returns None for an unreadable file."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.features.motion_features import extract_video_features

    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")

    feat = extract_video_features(bad)
    assert feat is None, "Expected None for unreadable video"


def test_short_video_returns_none(tmp_path, project_root):
    """extract_video_features returns None if < 2 sampled frames available."""
    import sys, cv2
    sys.path.insert(0, str(project_root))
    from src.features.motion_features import extract_video_features

    # Create video with only 1 frame, sample_interval=1 → only 1 sampled frame
    vid = tmp_path / "one_frame.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w = cv2.VideoWriter(str(vid), fourcc, 10.0, (64, 64))
    w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()

    feat = extract_video_features(vid, sample_interval=1, target_width=64, target_height=64)
    assert feat is None, "Expected None for video with only 1 sampled frame"
