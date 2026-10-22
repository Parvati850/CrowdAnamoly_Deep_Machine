"""
src/inference/alerts.py
------------------------
Alert generation and event logging for the real-time inference pipeline.

An alert fires only when ALL three conditions are true:
  1. Anomaly score >= threshold (note: equal-to ALSO triggers)
  2. This has persisted for `persistence` consecutive observations
  3. Cooldown since the last alert has elapsed

Each AlertEvent gets a unique ID so duplicates can be detected
and the Event Log remains traceable.
"""

import csv
import time
import uuid
from dataclasses import dataclass, field, asdict, fields as dc_fields
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AlertEvent:
    """Represents a single fired alert."""
    event_id:         str   # unique per event
    timestamp:        str
    source:           str
    score_abnormal:   float
    label:            str
    threshold:        float
    acknowledged:     bool = False
    dismissed:        bool = False
    notes:            str  = ""


# Ordered field names for CSV — kept stable so existing CSVs remain readable
_CSV_FIELDS = [f.name for f in dc_fields(AlertEvent)]


class AlertManager:
    """
    Stateful alert manager. Instantiate once per inference session.

    Parameters
    ----------
    threshold         : Anomaly score >= which a frame is "suspicious".
                        A score EQUAL to threshold also counts as suspicious.
    persistence       : Consecutive suspicious frames before alert fires.
    cooldown_seconds  : Minimum gap between successive alerts (seconds).
    log_path          : CSV file to append alert events (optional).
    source_name       : Human-readable source label used in all events
                        produced by this manager (e.g. the video filename).
    """

    def __init__(
        self,
        threshold: float = 0.55,
        persistence: int = 3,
        cooldown_seconds: float = 10.0,
        log_path: Optional[Path] = None,
        source_name: str = "unknown",
    ):
        self.threshold        = threshold
        self.persistence      = persistence
        self.cooldown_seconds = cooldown_seconds
        self.log_path         = Path(log_path) if log_path else None
        self.source_name      = source_name

        self._consecutive_high: int  = 0
        self._last_alert_time: float = 0.0
        self._events: list[AlertEvent] = []

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self._write_csv_header()

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        score: float,
        label: str,
        source: Optional[str] = None,
    ) -> Optional[AlertEvent]:
        """
        Feed one inference result.  Returns an AlertEvent if an alert fires,
        otherwise returns None.

        Parameters
        ----------
        score  : Anomaly score in [0, 1].
        label  : Model's predicted label string ("normal" | "abnormal").
        source : Override the source label for this specific event.
                 Falls back to self.source_name when not given.
        """
        effective_source = source if source is not None else self.source_name
        now = time.time()

        # >= so that score == threshold counts as suspicious
        if score >= self.threshold:
            self._consecutive_high += 1
        else:
            self._consecutive_high = 0
            return None

        if self._consecutive_high < self.persistence:
            return None

        if now - self._last_alert_time < self.cooldown_seconds:
            return None

        # All conditions met — fire
        self._last_alert_time  = now
        self._consecutive_high = 0

        event = AlertEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            source=effective_source,
            score_abnormal=round(score, 4),
            label=label,
            threshold=self.threshold,
        )
        self._events.append(event)
        self._append_to_csv(event)
        logger.warning(
            "ALERT fired | id=%s | source=%s | score=%.3f | threshold=%.3f",
            event.event_id, effective_source, score, self.threshold,
        )
        return event

    def acknowledge(self, event_id: str) -> bool:
        """Mark an event as acknowledged by its unique ID. Returns True if found."""
        for e in self._events:
            if e.event_id == event_id:
                e.acknowledged = True
                return True
        return False

    def dismiss(self, event_id: str) -> bool:
        """Mark an event as dismissed by its unique ID. Returns True if found."""
        for e in self._events:
            if e.event_id == event_id:
                e.dismissed = True
                return True
        return False

    def reset(self) -> None:
        """Reset the consecutive counter (call when switching video sources)."""
        self._consecutive_high = 0

    @property
    def events(self) -> list[AlertEvent]:
        return list(self._events)

    @property
    def active_count(self) -> int:
        return sum(1 for e in self._events if not e.dismissed)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _write_csv_header(self) -> None:
        with open(self.log_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writeheader()

    def _append_to_csv(self, event: AlertEvent) -> None:
        if self.log_path is None:
            return
        with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writerow(asdict(event))
