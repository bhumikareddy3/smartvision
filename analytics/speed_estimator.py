"""
Speed Estimator
===============
Estimates object speed from pixel displacement between frames.

Formula:
  speed_px/frame = distance(pos_t, pos_t-1)
  speed_m/s      = (speed_px/frame) * fps / pixels_per_meter
  speed_km/h     = speed_m/s * 3.6

Calibration:
  pixels_per_meter must be calibrated per scene. A known reference object
  (e.g. a lane marking of known width) provides the px-to-metre ratio.
"""

from collections import defaultdict, deque
from typing import Dict, Optional, List
import numpy as np


class SpeedEstimator:
    """
    Parameters
    ----------
    pixels_per_meter : float
        Pixels in the image that correspond to 1 real-world metre.
    fps              : float
        Video frames per second (used to convert px/frame → m/s).
    window           : int
        Rolling average window (frames) to smooth speed estimates.
    """

    def __init__(
        self,
        pixels_per_meter: float = 15.0,
        fps: float = 30.0,
        window: int = 5,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.fps              = fps
        self.window           = window
        self._speed_buffers:  Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_positions: Dict[int, tuple]  = {}

    def update(self, track_list: List[dict]) -> List[dict]:
        """
        Calculate instantaneous speed for each track and annotate the dict
        with key "speed_kmh".
        """
        for t in track_list:
            tid = t["track_id"]
            cx, cy = t["cx"], t["cy"]

            if tid in self._last_positions:
                lx, ly = self._last_positions[tid]
                pixel_dist = np.hypot(cx - lx, cy - ly)
                speed_ms   = pixel_dist * self.fps / self.pixels_per_meter
                speed_kmh  = speed_ms * 3.6
                self._speed_buffers[tid].append(speed_kmh)

            self._last_positions[tid] = (cx, cy)

            # Smoothed speed
            buf = self._speed_buffers[tid]
            t["speed_kmh"] = float(np.mean(buf)) if buf else 0.0

        return track_list

    def get_speed(self, track_id: int) -> float:
        buf = self._speed_buffers.get(track_id)
        return float(np.mean(buf)) if buf else 0.0

    def update_fps(self, fps: float):
        self.fps = max(1.0, fps)

    def reset(self):
        self._speed_buffers.clear()
        self._last_positions.clear()
