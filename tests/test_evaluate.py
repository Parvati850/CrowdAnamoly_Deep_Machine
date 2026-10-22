"""
tests/test_evaluate.py
-----------------------
Tests for evaluation metric computation.
"""

import pytest
import numpy as np


def test_metrics_perfect_prediction(project_root):
    """Perfect predictions yield balanced_accuracy=1.0 and roc_auc=1.0."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.evaluate import compute_metrics

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    probs  = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])

    m = compute_metrics(y_true, y_pred, probs)
    assert m["balanced_accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_metrics_all_wrong(project_root):
    """All-wrong predictions yield balanced_accuracy=0.0."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.evaluate import compute_metrics

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    m = compute_metrics(y_true, y_pred)
    assert m["balanced_accuracy"] == 0.0


def test_metrics_empty_set_returns_error(project_root):
    """Empty test set returns an error dict instead of crashing."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.evaluate import compute_metrics

    m = compute_metrics(np.array([]), np.array([]))
    assert "error" in m


def test_fp_fn_counts(project_root):
    """FP and FN counts are correct."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.evaluate import compute_metrics

    # 1 FP (predicted abnormal, actually normal), 1 FN (predicted normal, actually abnormal)
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1])

    m = compute_metrics(y_true, y_pred)
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tp"] == 1
    assert m["tn"] == 1


def test_roc_auc_single_class_is_none(project_root):
    """ROC-AUC returns None when only one class is present."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.models.evaluate import compute_metrics

    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    probs  = np.array([[0.9, 0.1]] * 3)

    m = compute_metrics(y_true, y_pred, probs)
    assert m["roc_auc"] is None
