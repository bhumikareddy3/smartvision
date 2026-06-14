"""
Object Tracker Wrapper
======================
Thin state-management layer on top of the ultralytics ByteTrack integration.

Responsibilities:
  • Maintain per-track position history (needed for speed estimation)
  • Detect when a track is "new" vs "existing"
  • Store last-seen positions for line-crossing logic
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import numpy as np


class TrackState:
    """Holds rolling history for a single tracked object."""

    def __init__(self, track_id: int, maxlen: int = 30):
        self.track_id = track_id
        self.positions: deque = deque(maxlen=maxlen)   # (cx, cy) tuples
        self.class_id: Optional[int] = None
        self.class_name: str = ""
        self.confidence: float = 0.0
        self.last_bbox: Optional[Tuple] = None
        self.frame_count: int = 0
        self.is_new: bool = True

    def update(self, track_dict: dict):
        self.positions.append((track_dict["cx"], track_dict["cy"]))
        self.class_id   = track_dict["class_id"]
        self.class_name = track_dict["class_name"]
        self.confidence = track_dict["confidence"]
        self.last_bbox  = (
            track_dict["x1"], track_dict["y1"],
            track_dict["x2"], track_dict["y2"],
        )
        self.frame_count += 1
        if self.frame_count > 1:
            self.is_new = False

    @property
    def current_pos(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    @property
    def prev_pos(self) -> Optional[Tuple[float, float]]:
        return self.positions[-2] if len(self.positions) >= 2 else None


class ObjectTracker:
    """
    Manages the dictionary of active TrackState objects.

    This is a pure state layer — actual ByteTrack assignment
    is done by ultralytics inside ObjectDetector.track().
    """

    def __init__(self, history_len: int = 30):
        self.history_len = history_len
        self._states: Dict[int, TrackState] = {}
        self._lost_ids: set = set()

    def update(self, track_list: List[dict]) -> List[dict]:
        """
        Ingest a list of track dicts (from ObjectDetector.parse_tracks),
        update internal state, and annotate each dict with:
          • is_new  : bool — first time we've seen this ID
          • positions : list of (cx, cy) history

        Returns the annotated list.
        """
        active_ids = set()

        for t in track_list:
            tid = t["track_id"]
            active_ids.add(tid)

            if tid not in self._states:
                self._states[tid] = TrackState(tid, self.history_len)

            state = self._states[tid]
            state.update(t)

            # Annotate dict in place
            t["is_new"]    = state.is_new
            t["positions"] = list(state.positions)

        # Mark IDs not seen this frame as lost
        current_ids = set(self._states.keys())
        self._lost_ids = current_ids - active_ids

        return track_list

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self._states.get(track_id)

    def get_position_history(self, track_id: int) -> List[Tuple]:
        state = self._states.get(track_id)
        return list(state.positions) if state else []

    def get_prev_position(self, track_id: int) -> Optional[Tuple]:
        state = self._states.get(track_id)
        return state.prev_pos if state else None

    def active_count(self) -> int:
        return len(self._states) - len(self._lost_ids)

    def reset(self):
        self._states.clear()
        self._lost_ids.clear()
