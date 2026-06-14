"""
Alert Manager
=============
Fires alerts when configurable thresholds are exceeded.

Alert types:
  OCCUPANCY  — too many objects in the scene
  DENSITY    — crowd density exceeds threshold
  SPEED      — a vehicle is speeding

Cooldown logic prevents alert spam: the same alert type can only fire
once per `cooldown_seconds` window.
"""

import time
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Alert:
    alert_type: str
    message:    str
    severity:   str
    timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "message":    self.message,
            "severity":   self.severity,
            "timestamp":  self.timestamp,
        }


class AlertManager:
    """
    Parameters
    ----------
    occupancy_threshold : int
        Max objects in scene before OCCUPANCY alert fires.
    density_threshold   : float
        Crowd density [0–1] before DENSITY alert fires.
    speed_threshold_kmh : float
        Speed in km/h before SPEED alert fires.
    cooldown_seconds    : float
        Minimum seconds between repeated alerts of the same type.
    """

    def __init__(
        self,
        occupancy_threshold: int   = 20,
        density_threshold:   float = 0.65,
        speed_threshold_kmh: float = 80.0,
        cooldown_seconds:    float = 30.0,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ):
        self.occupancy_threshold = occupancy_threshold
        self.density_threshold   = density_threshold
        self.speed_threshold_kmh = speed_threshold_kmh
        self.cooldown_seconds    = cooldown_seconds
        self.on_alert            = on_alert  # optional callback

        self._last_fire:   Dict[str, float] = {}
        self._alert_log:   List[Alert]       = []

    # ── Check helpers ─────────────────────────────────────────────────────────

    def check_occupancy(self, count: int) -> Optional[Alert]:
        if count >= self.occupancy_threshold:
            return self._fire(
                "OCCUPANCY",
                f"High occupancy: {count} objects detected (threshold {self.occupancy_threshold})",
                "WARNING",
            )
        return None

    def check_density(self, density: float, person_count: int) -> Optional[Alert]:
        if density >= self.density_threshold:
            return self._fire(
                "DENSITY",
                f"High crowd density: {density:.1%} ({person_count} persons)",
                "CRITICAL",
            )
        return None

    def check_speed(self, track_id: int, class_name: str, speed_kmh: float) -> Optional[Alert]:
        if speed_kmh >= self.speed_threshold_kmh:
            return self._fire(
                f"SPEED_{track_id}",
                f"Speeding {class_name} (ID {track_id}): {speed_kmh:.1f} km/h",
                "WARNING",
                cooldown_key="SPEED",   # group all speed alerts under one cooldown
            )
        return None

    def process_all(
        self,
        occupancy: int,
        density: float,
        person_count: int,
        track_list: List[dict],
    ) -> List[Alert]:
        """Run all checks and return newly fired alerts."""
        fired = []
        a = self.check_occupancy(occupancy)
        if a:
            fired.append(a)
        a = self.check_density(density, person_count)
        if a:
            fired.append(a)
        for t in track_list:
            spd = t.get("speed_kmh", 0)
            if spd > 0:
                a = self.check_speed(t["track_id"], t["class_name"], spd)
                if a:
                    fired.append(a)
        return fired

    @property
    def recent_alerts(self) -> List[Alert]:
        """Return last 50 alerts in reverse-chronological order."""
        return list(reversed(self._alert_log[-50:]))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fire(
        self,
        alert_type: str,
        message: str,
        severity: str,
        cooldown_key: Optional[str] = None,
    ) -> Optional[Alert]:
        key = cooldown_key or alert_type
        now = time.monotonic()
        if now - self._last_fire.get(key, 0) < self.cooldown_seconds:
            return None

        self._last_fire[key] = now
        alert = Alert(alert_type=alert_type, message=message, severity=severity)
        self._alert_log.append(alert)
        if self.on_alert:
            self.on_alert(alert)
        return alert
