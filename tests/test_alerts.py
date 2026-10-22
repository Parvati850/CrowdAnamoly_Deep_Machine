"""
tests/test_alerts.py
---------------------
Tests for the AlertManager threshold, persistence, cooldown, unique IDs,
source_name, and ack/dismiss logic.
"""

import pytest
import time
from pathlib import Path


# ── Basic threshold logic ────────────────────────────────────────────────────

def test_no_alert_below_threshold(project_root):
    """No alert fires when score is strictly below threshold."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.6, persistence=1, cooldown_seconds=0)
    assert mgr.update(0.59, "normal") is None


def test_alert_fires_at_threshold_boundary(project_root):
    """Score exactly equal to threshold is treated as suspicious (>=)."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.6, persistence=1, cooldown_seconds=0)
    event = mgr.update(0.6, "abnormal")   # exactly at threshold → should fire
    assert event is not None, "Score == threshold should trigger an alert"


def test_alert_fires_above_threshold_with_persistence(project_root):
    """Alert fires only after persistence consecutive high-score updates."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.6, persistence=3, cooldown_seconds=0)
    assert mgr.update(0.8, "abnormal") is None   # 1st — not yet
    assert mgr.update(0.8, "abnormal") is None   # 2nd — not yet
    event = mgr.update(0.8, "abnormal")           # 3rd — fires
    assert event is not None
    assert event.score_abnormal == 0.8


def test_alert_resets_on_low_score(project_root):
    """Consecutive counter resets when score drops below threshold."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.6, persistence=3, cooldown_seconds=0)
    mgr.update(0.8, "abnormal")   # 1
    mgr.update(0.8, "abnormal")   # 2
    mgr.update(0.3, "normal")     # drops below — counter resets
    result = mgr.update(0.8, "abnormal")   # only 1 consecutive → no alert
    assert result is None


def test_cooldown_prevents_rapid_repeat(project_root):
    """Cooldown prevents a second alert firing immediately after the first."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=60)
    e1 = mgr.update(0.9, "abnormal")
    assert e1 is not None
    e2 = mgr.update(0.9, "abnormal")
    assert e2 is None, "Second alert should be blocked by cooldown"


# ── Unique event IDs ─────────────────────────────────────────────────────────

def test_each_event_has_unique_id(project_root):
    """Every fired alert gets a non-empty unique event_id."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=0)
    e1 = mgr.update(0.9, "abnormal")
    # Reset cooldown simulation by bumping _last_alert_time
    mgr._last_alert_time = 0.0
    e2 = mgr.update(0.9, "abnormal")
    assert e1 is not None and e2 is not None
    assert e1.event_id != e2.event_id, "Event IDs must be unique"
    assert len(e1.event_id) > 0


# ── source_name ──────────────────────────────────────────────────────────────

def test_source_name_used_in_events(project_root):
    """AlertManager uses source_name as the event source."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=0,
                       source_name="my_video.avi")
    e = mgr.update(0.9, "abnormal")
    assert e is not None
    assert e.source == "my_video.avi"


def test_source_override_per_call(project_root):
    """Per-call source kwarg overrides source_name."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=0,
                       source_name="default_source")
    e = mgr.update(0.9, "abnormal", source="override_source")
    assert e is not None
    assert e.source == "override_source"


# ── ack / dismiss ────────────────────────────────────────────────────────────

def test_acknowledge_by_event_id(project_root):
    """acknowledge() sets acknowledged=True on the correct event."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=0)
    event = mgr.update(0.9, "abnormal")
    assert event is not None
    result = mgr.acknowledge(event.event_id)
    assert result is True
    assert mgr.events[0].acknowledged is True


def test_dismiss_by_event_id(project_root):
    """dismiss() sets dismissed=True and reduces active_count."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=0)
    event = mgr.update(0.9, "abnormal")
    assert event is not None
    assert mgr.active_count == 1
    mgr.dismiss(event.event_id)
    assert mgr.active_count == 0
    assert mgr.events[0].dismissed is True


def test_acknowledge_unknown_id_returns_false(project_root):
    """acknowledge() with an unknown ID returns False without crashing."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=0)
    result = mgr.acknowledge("nonexistent-id")
    assert result is False


# ── Events accessible ────────────────────────────────────────────────────────

def test_events_logged_and_countable(project_root):
    """Fired events are accessible via .events and .active_count."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=0)
    mgr.update(0.9, "abnormal")
    assert len(mgr.events) == 1
    assert mgr.active_count == 1


def test_csv_written(tmp_path, project_root):
    """Alert events are written to a CSV file and contain expected fields."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    log = tmp_path / "alerts.csv"
    mgr = AlertManager(threshold=0.5, persistence=1, cooldown_seconds=0,
                       log_path=log, source_name="test_video.avi")
    mgr.update(0.9, "abnormal")
    assert log.exists()
    content = log.read_text()
    assert "abnormal" in content
    assert "event_id" in content     # header present
    assert "test_video.avi" in content


def test_no_duplicate_events_from_single_fire(project_root):
    """One persistence-trigger cycle produces exactly one event, not multiples."""
    import sys; sys.path.insert(0, str(project_root))
    from src.inference.alerts import AlertManager
    mgr = AlertManager(threshold=0.5, persistence=3, cooldown_seconds=0)
    for _ in range(3):
        mgr.update(0.9, "abnormal")
    # Only one event should have been created
    assert len(mgr.events) == 1
