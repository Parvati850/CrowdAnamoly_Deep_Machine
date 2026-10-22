"""
tests/test_alert_manager_constructor.py
----------------------------------------
Regression tests for the AlertManager constructor.

The original bug: app.py called AlertManager(..., source_name=uploaded.name)
but an old cached version of alerts.py lacked the source_name parameter,
causing:
    TypeError: AlertManager.__init__() got an unexpected keyword argument 'source_name'

These tests guard against that regression and verify the full constructor API
matches every call pattern used in app.py, realtime.py, and the test suite.
"""

import pytest
from pathlib import Path


# ── Constructor compatibility ────────────────────────────────────────────────

def test_constructor_accepts_source_name(project_root):
    """AlertManager must accept source_name as a keyword argument."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager

    # This is the exact call pattern from app.py Video Analysis page
    mgr = AlertManager(
        threshold=0.55,
        persistence=3,
        cooldown_seconds=10.0,
        source_name="861-13_70.mov",
    )
    assert mgr.source_name == "861-13_70.mov"
    assert mgr.threshold == 0.55
    assert mgr.persistence == 3


def test_constructor_default_source_name(project_root):
    """source_name defaults to 'unknown' when not provided."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager

    mgr = AlertManager(threshold=0.55, persistence=1, cooldown_seconds=0)
    assert mgr.source_name == "unknown"


def test_constructor_with_log_path(project_root, tmp_path):
    """Constructor accepts log_path alongside source_name."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager

    log = tmp_path / "alerts.csv"
    mgr = AlertManager(
        threshold=0.6,
        persistence=1,
        cooldown_seconds=0,
        log_path=log,
        source_name="test_video.avi",
    )
    assert mgr.source_name == "test_video.avi"
    assert mgr.log_path == log


def test_constructor_exact_app_py_video_analysis_pattern(project_root):
    """
    Reproduces the exact app.py call pattern that caused:
        TypeError: AlertManager.__init__() got an unexpected keyword argument 'source_name'
    """
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    from src.utils.paths import load_config

    cfg = load_config()

    # Exact replica of app.py ~line 329
    try:
        alert_mgr = AlertManager(
            threshold=0.55,                          # from sidebar slider
            persistence=cfg["alerts"]["persistence"],
            cooldown_seconds=cfg["alerts"]["cooldown_seconds"],
            source_name="861-13_70.mov",             # uploaded.name
        )
    except TypeError as e:
        pytest.fail(
            f"AlertManager constructor raised TypeError: {e}\n"
            "This is the regression that was reported. "
            "Check that alerts.py contains source_name in __init__."
        )

    assert alert_mgr is not None
    assert alert_mgr.source_name == "861-13_70.mov"


def test_constructor_exact_live_monitor_pattern(project_root):
    """Reproduces the app.py Live Monitor AlertManager call pattern."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    from src.utils.paths import load_config

    cfg = load_config()
    source_label = "Webcam 0"

    alert_mgr = AlertManager(
        threshold=0.55,
        persistence=cfg["alerts"]["persistence"],
        cooldown_seconds=cfg["alerts"]["cooldown_seconds"],
        source_name=source_label,
    )
    assert alert_mgr.source_name == source_label


# ── Source name flows into events ────────────────────────────────────────────

def test_source_name_appears_in_fired_event(project_root):
    """source_name set at construction appears in the fired AlertEvent."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager

    mgr = AlertManager(
        threshold=0.5, persistence=1, cooldown_seconds=0,
        source_name="my_crowd_video.mp4",
    )
    event = mgr.update(0.9, "abnormal")
    assert event is not None
    assert event.source == "my_crowd_video.mp4"


def test_source_name_not_numeric_string(project_root):
    """source_name must not silently become '0' (the old webcam-int bug)."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager

    mgr = AlertManager(
        threshold=0.5, persistence=1, cooldown_seconds=0,
        source_name="Webcam 0",          # correct: human-readable label
    )
    event = mgr.update(0.9, "abnormal")
    assert event is not None
    assert event.source == "Webcam 0"
    assert event.source != "0"          # was the old bug


# ── Inspect signature directly ────────────────────────────────────────────────

def test_init_signature_contains_source_name(project_root):
    """Directly inspect __init__ to confirm source_name is in the signature."""
    import sys, inspect; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager

    params = list(inspect.signature(AlertManager.__init__).parameters.keys())
    assert "source_name" in params, (
        f"'source_name' not found in AlertManager.__init__ params: {params}\n"
        "The runtime error was caused by an outdated cached .pyc. "
        "Ensure alerts.py is saved and the server is restarted."
    )
