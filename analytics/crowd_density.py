"""
Crowd Density Monitor
=====================
Divides the frame into a grid and measures what fraction of cells
contain at least one person, producing a density score [0, 1].
"""

from typing import List, Tuple
import numpy as np


class CrowdDensityMonitor:
    """
    Parameters
    ----------
    frame_size : (height, width) of the video frame
    grid_rows  : Number of grid rows
    grid_cols  : Number of grid columns
    """

    def __init__(
        self,
        frame_size: Tuple[int, int],
        grid_rows: int = 8,
        grid_cols: int = 8,
    ):
        self.frame_size = frame_size
        self.grid_rows  = grid_rows
        self.grid_cols  = grid_cols
        self._grid      = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, track_list: List[dict]):
        """Populate the density grid from the current track list."""
        self._grid[:] = 0
        h, w = self.frame_size
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols

        for t in track_list:
            # Only count persons (class_id 0)
            if t["class_id"] != 0:
                continue
            row = int(t["cy"] / cell_h)
            col = int(t["cx"] / cell_w)
            row = min(row, self.grid_rows - 1)
            col = min(col, self.grid_cols - 1)
            self._grid[row, col] += 1

    @property
    def density(self) -> float:
        """Fraction of cells that contain ≥1 person."""
        occupied = np.count_nonzero(self._grid)
        return occupied / (self.grid_rows * self.grid_cols)

    @property
    def person_count(self) -> int:
        return int(self._grid.sum())

    @property
    def grid(self) -> np.ndarray:
        return self._grid.copy()

    def level(self) -> str:
        d = self.density
        if d < 0.2:
            return "Low"
        elif d < 0.5:
            return "Medium"
        elif d < 0.75:
            return "High"
        else:
            return "Critical"

    def resize(self, new_size: Tuple[int, int]):
        self.frame_size = new_size
