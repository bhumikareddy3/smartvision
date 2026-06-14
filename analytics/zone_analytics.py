"""
Zone-Based Analytics
====================
Monitors user-defined rectangular regions of interest (ROI).

Each zone tracks:
  • Current occupancy (objects inside the zone right now)
  • Entry / exit events per track ID
  • Per-class occupancy breakdown
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Zone:
    """A single rectangular monitoring zone."""
    id:     int
    name:   str
    x1:     float
    y1:     float
    x2:     float
    y2:     float
    color:  Tuple[int, int, int] = (0, 200, 255)

    # Runtime state (not persisted)
    current_ids:    set         = field(default_factory=set, repr=False)
    class_counts:   Dict        = field(default_factory=lambda: defaultdict(int), repr=False)
    total_entered:  int         = 0
    total_exited:   int         = 0

    def contains(self, cx: float, cy: float) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    @property
    def occupancy(self) -> int:
        return len(self.current_ids)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class ZoneAnalytics:
    """
    Manages a collection of zones and processes track positions against them.

    Usage:
        za = ZoneAnalytics()
        za.add_zone(Zone(id=0, name="Entrance", x1=0, y1=0, x2=300, y2=400))
        events = za.process_tracks(track_list)
        # events is a list of dicts with zone_id, track_id, event_type
    """

    def __init__(self):
        self._zones: Dict[int, Zone] = {}
        self._track_zone_membership: Dict[int, set] = defaultdict(set)

    # ── Zone management ───────────────────────────────────────────────────────

    def add_zone(self, zone: Zone):
        self._zones[zone.id] = zone

    def remove_zone(self, zone_id: int):
        self._zones.pop(zone_id, None)

    def clear_zones(self):
        self._zones.clear()
        self._track_zone_membership.clear()

    def get_zones(self) -> List[Zone]:
        return list(self._zones.values())

    def update_zone(self, zone_id: int, **kwargs):
        if zone_id in self._zones:
            z = self._zones[zone_id]
            for k, v in kwargs.items():
                if hasattr(z, k):
                    setattr(z, k, v)

    # ── Track processing ──────────────────────────────────────────────────────

    def process_tracks(self, track_list: List[dict]) -> List[dict]:
        """
        For each track, determine which zones it's inside.
        Detect ENTER / EXIT transitions.

        Returns list of event dicts:
          {zone_id, zone_name, track_id, object_type, event_type}
        """
        events = []
        active_tids = {t["track_id"] for t in track_list}

        # ── Handle exits for tracks no longer active ──────────────────────────
        gone_tids = set(self._track_zone_membership.keys()) - active_tids
        for tid in gone_tids:
            for zid in list(self._track_zone_membership[tid]):
                zone = self._zones.get(zid)
                if zone:
                    zone.current_ids.discard(tid)
                    zone.total_exited += 1
                    events.append(dict(
                        zone_id=zid, zone_name=zone.name,
                        track_id=tid, object_type="unknown",
                        event_type="EXIT",
                    ))
            del self._track_zone_membership[tid]

        # ── Evaluate current tracks ───────────────────────────────────────────
        for t in track_list:
            tid  = t["track_id"]
            cx   = t["cx"]
            cy   = t["cy"]
            name = t["class_name"]

            current_zones = set()
            for zid, zone in self._zones.items():
                if zone.contains(cx, cy):
                    current_zones.add(zid)

            prev_zones = self._track_zone_membership.get(tid, set())

            # ENTER events
            for zid in current_zones - prev_zones:
                zone = self._zones[zid]
                zone.current_ids.add(tid)
                zone.class_counts[name] += 1
                zone.total_entered += 1
                events.append(dict(
                    zone_id=zid, zone_name=zone.name,
                    track_id=tid, object_type=name,
                    event_type="ENTER",
                ))

            # EXIT events
            for zid in prev_zones - current_zones:
                zone = self._zones[zid]
                zone.current_ids.discard(tid)
                zone.total_exited += 1
                events.append(dict(
                    zone_id=zid, zone_name=zone.name,
                    track_id=tid, object_type=name,
                    event_type="EXIT",
                ))

            self._track_zone_membership[tid] = current_zones

        return events

    # ── Summaries ─────────────────────────────────────────────────────────────

    def get_occupancy_summary(self) -> Dict[str, int]:
        return {z.name: z.occupancy for z in self._zones.values()}

    def total_occupancy(self) -> int:
        return sum(z.occupancy for z in self._zones.values())
