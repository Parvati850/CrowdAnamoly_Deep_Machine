"""
tests/test_model.py
--------------------
Tests for baseline model: training, saving, loading, and inference.
Uses only synthetic numpy arrays — no real videos or files needed.
"""

import pytest
import numpy as np
import pickle
from pathlib import Path


@pytest.fixture
def synthetic_xy():
    """4 normal + 4 abnormal synthetic feature vectors."""
    rng = np.random.default_rng(0)
    X = rng.random((8, 33)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    return X, y


@pytest.fixture
def minimal_cfg():
    return {
        "model": {
            "baseline_classifier": "random_forest",
            "n_estimators": 10,
            "max_depth": 3,
        },
        "training": {"random_seed": 42},
    }


def test_model_trains(synthetic_xy, minimal_cfg, project_root):
    """Model can be trained without error."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.baseline import train

    X, y = synthetic_xy
    pipeline = train(X, y, minimal_cfg)
    assert pipeline is not None


def test_model_predict_shape(synthetic_xy, minimal_cfg, project_root):
    """Predict returns correct shapes."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.baseline import train, predict

    X, y = synthetic_xy
    pipeline = train(X, y, minimal_cfg)
    labels, probs = predict(pipeline, X)

    assert labels.shape == (8,)
    assert probs.shape == (8, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_model_save_load(tmp_path, synthetic_xy, minimal_cfg, project_root):
    """Model can be serialized and reloaded."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.baseline import train, save_model, load_model, predict

    X, y = synthetic_xy
    pipeline = train(X, y, minimal_cfg)

    ckpt = tmp_path / "model.pkl"
    save_model(pipeline, ckpt)
    assert ckpt.exists()

    reloaded = load_model(ckpt)
    labels1, probs1 = predict(pipeline, X)
    labels2, probs2 = predict(reloaded, X)

    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_allclose(probs1, probs2, atol=1e-6)


def test_load_missing_model_raises(tmp_path, project_root):
    """load_model raises FileNotFoundError for missing checkpoint."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.baseline import load_model

    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.pkl")


def test_empty_train_raises(minimal_cfg, project_root):
    """train() raises ValueError on empty training set."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.baseline import train

    X = np.empty((0, 33), dtype=np.float32)
    y = np.empty((0,), dtype=int)

    with pytest.raises(ValueError):
        train(X, y, minimal_cfg)


def test_different_classifiers(synthetic_xy, project_root):
    """All three supported classifier types train without error."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.baseline import train, predict

    X, y = synthetic_xy
    for clf_name in ["random_forest", "svm", "logistic"]:
        cfg = {
            "model": {"baseline_classifier": clf_name, "n_estimators": 5, "max_depth": 2},
            "training": {"random_seed": 42},
        }
        pipeline = train(X, y, cfg)
        labels, probs = predict(pipeline, X)
        assert len(labels) == 8, f"{clf_name} prediction length mismatch"
