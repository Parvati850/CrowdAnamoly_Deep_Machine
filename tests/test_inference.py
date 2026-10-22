"""
tests/test_inference.py
------------------------
Tests for video inference pipeline.
Uses the synthetic_video fixture — no original files used.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def trained_model(synthetic_video, project_root):
    """Train a tiny model on two synthetic feature vectors for inference tests."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.features.motion_features import extract_video_features
    from src.models.baseline import train

    feat = extract_video_features(synthetic_video, sample_interval=2,
                                  target_width=64, target_height=64)
    assert feat is not None

    # Two samples: same vector for both classes (model will learn something trivial)
    X = np.stack([feat, feat * 1.1])
    y = np.array([0, 1])

    cfg = {
        "model": {"baseline_classifier": "random_forest",
                  "n_estimators": 5, "max_depth": 2},
        "training": {"random_seed": 42},
    }
    return train(X, y, cfg)


def test_predict_video_with_model(synthetic_video, trained_model, project_root):
    """predict_video returns a valid result dict for a readable video."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.predict import predict_video
    from src.utils.paths import load_config

    cfg = load_config()
    result = predict_video(synthetic_video, cfg=cfg, model=trained_model)

    assert result["feature_ok"] is True
    assert result["model_loaded"] is True
    assert result["error"] is None
    assert result["label"] in ("normal", "abnormal")
    assert 0.0 <= result["score_abnormal"] <= 1.0
    assert abs(result["score_normal"] + result["score_abnormal"] - 1.0) < 1e-4


def test_predict_video_missing_file(trained_model, tmp_path, project_root):
    """predict_video returns an error for a non-existent file."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.predict import predict_video
    from src.utils.paths import load_config

    cfg = load_config()
    result = predict_video(tmp_path / "nonexistent.avi", cfg=cfg, model=trained_model)

    assert result["error"] is not None
    assert result["label"] == "error"


def test_analyse_video_returns_windows(synthetic_video, trained_model, project_root):
    """analyse_video returns window-level scores for a readable video."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.inference.video_inference import analyse_video
    from src.utils.paths import load_config

    cfg = load_config()
    result = analyse_video(
        synthetic_video, cfg=cfg, model=trained_model, window_size=5
    )

    assert result["error"] is None, f"analyse_video error: {result['error']}"
    # With 20 frames at default interval=10, we get 2 sampled frames
    # window_size=5 → not enough for a window; that's fine, result should have 0 windows
    assert isinstance(result["windows"], list)
    assert result["fps_processing"] > 0
