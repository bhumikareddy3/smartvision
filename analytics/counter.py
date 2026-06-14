"""
Line Crossing Counter
=====================
Detects when a tracked object crosses a user-defined virtual line.

Logic:
  - The line spans the full width of the frame at a configurable Y position.
  - Each track's PREVIOUS vs CURRENT Y-centre is compared each frame.
  - A crossing is logged at most once per track (duplicate prevention via a set).
  - Direction:
      cy_prev > line_y AND cy_curr <= line_y  →  "IN"  (moving upward / towards camera)
      cy_prev < line_y AND cy_curr >= line_y  →  "OUT" (moving downward / away)
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class LineCounter:
    """
    Virtual counting line.

    Parameters
    ----------
    line_y : float
        Absolute pixel Y-coordinate of the line.
    frame_width : int
        Width of the video frame (used only for drawing, not logic).
    tolerance : int
        Extra pixels of tolerance around the line (reduces jitter-bouncing).
    """

    def __init__(
        self,
        line_y: float,
        frame_width: int = 1280,
        tolerance: int = 5,
    ):
        self.line_y       = float(line_y)
        self.frame_width  = frame_width
        self.tolerance    = tolerance

        # Counts
        self.count_in: int  = 0
        self.count_out: int = 0

        # Per-class counts
        self.class_count_in:  Dict[str, int] = defaultdict(int)
        self.class_count_out: Dict[str, int] = defaultdict(int)

        # Tracks that have already crossed (prevent double-counting)
        self._crossed_ids: set = set()

        # Last recorded side for each track: "above" | "below"
        self._track_side: Dict[int, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def set_line(self, line_y: float, frame_width: int):
        """Reposition the line (e.g. when user drags slider)."""
        self.line_y      = float(line_y)
        self.frame_width = frame_width

    def process_tracks(self, track_list: List[dict]) -> List[dict]:
        """
        Evaluate each track for a crossing event.

        Adds key "crossing" → None | "IN" | "OUT" to each dict.
        Returns the annotated list.
        """
        for t in track_list:
            t["crossing"] = self._check_crossing(t)
        return track_list

    def reset_counts(self):
        self.count_in  = 0
        self.count_out = 0
        self.class_count_in.clear()
        self.class_count_out.clear()
        self._crossed_ids.clear()
        self._track_side.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _side(self, cy: float) -> str:
        """Return 'above' if cy < line_y, else 'below'."""
        return "above" if cy < self.line_y else "below"

    def _check_crossing(self, t: dict) -> Optional[str]:
        tid  = t["track_id"]
        cy   = t["cy"]
        name = t["class_name"]

        current_side = self._side(cy)
        prev_side    = self._track_side.get(tid)

        # Update side record
        self._track_side[tid] = current_side

        # No previous position yet or hasn't switched sides
        if prev_side is None or prev_side == current_side:
            return None

        # Already counted once — prevent duplicates
        if tid in self._crossed_ids:
            return None

        # Determine direction
        # "above→below" = moving down = OUT
        # "below→above" = moving up   = IN
        if prev_side == "above" and current_side == "below":
            direction = "OUT"
            self.count_out += 1
            self.class_count_out[name] += 1
        else:
            direction = "IN"
            self.count_in  += 1
            self.class_count_in[name]  += 1

        self._crossed_ids.add(tid)
        return direction

    # ── Line geometry (for drawing) ───────────────────────────────────────────

    @property
    def line_start(self) -> Tuple[int, int]:
        return (0, int(self.line_y))

    @property
    def line_end(self) -> Tuple[int, int]:
        return (self.frame_width, int(self.line_y))
