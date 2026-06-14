"""
Heatmap Generator
=================
Accumulates object centre-point positions into a 2-D intensity map
and blends it with the live video frame.

A decay factor is applied each frame so old positions fade over time,
giving a "recent movement" heatmap rather than an all-time accumulation.
"""

from typing import Tuple, List
import numpy as np
import cv2


class HeatmapGenerator:
    """
    Parameters
    ----------
    frame_size : (height, width)
    decay      : Multiplier applied to the heatmap each frame (0–1).
                 Higher = slower decay = longer memory.
    alpha      : Blend weight of the heatmap overlay (0 = invisible, 1 = opaque).
    """

    def __init__(
        self,
        frame_size: Tuple[int, int],
        decay: float = 0.98,
        alpha: float = 0.5,
    ):
        self.decay  = decay
        self.alpha  = alpha
        h, w = frame_size
        # Float32 accumulator — one channel
        self._map = np.zeros((h, w), dtype=np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, track_list: List[dict]):
        """Stamp a Gaussian blob at each object's centre."""
        # Apply per-frame decay first
        self._map *= self.decay

        for t in track_list:
            cx = int(t["cx"])
            cy = int(t["cy"])
            self._stamp(cx, cy)

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the colourised heatmap onto `frame` and return the result.
        Does NOT modify `frame` in place.
        """
        if self._map.max() == 0:
            return frame

        # Normalise to 0–255
        norm = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        # Apply COLORMAP_JET: cold→hot = blue→red
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Mask: only blend where there is actually heat
        mask = norm > 10
        coloured[~mask] = 0

        blended = frame.copy()
        blended[mask] = cv2.addWeighted(
            frame[mask], 1 - self.alpha,
            coloured[mask], self.alpha, 0,
        )
        return blended

    def reset(self):
        self._map[:] = 0

    def resize(self, new_size: Tuple[int, int]):
        """Resize accumulator when frame dimensions change."""
        h, w = new_size
        self._map = np.zeros((h, w), dtype=np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _stamp(self, cx: int, cy: int, radius: int = 30, intensity: float = 1.0):
        """Add a soft Gaussian blob centred at (cx, cy)."""
        h, w = self._map.shape
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return

        # Build a small Gaussian kernel and add it to the map at the right location
        size    = radius * 2 + 1
        sigma   = radius / 3.0
        kernel  = cv2.getGaussianKernel(size, sigma)
        kernel2d = kernel @ kernel.T
        kernel2d = (kernel2d / kernel2d.max()) * intensity

        # Compute clipped slice indices
        x0, x1 = cx - radius, cx + radius + 1
        y0, y1 = cy - radius, cy + radius + 1

        kx0 = max(0, -x0);  kx1 = kx0 + (min(x1, w) - max(x0, 0))
        ky0 = max(0, -y0);  ky1 = ky0 + (min(y1, h) - max(y0, 0))
        ix0 = max(x0, 0);   ix1 = min(x1, w)
        iy0 = max(y0, 0);   iy1 = min(y1, h)

        if kx1 > kx0 and ky1 > ky0:
            self._map[iy0:iy1, ix0:ix1] += kernel2d[ky0:ky1, kx0:kx1]
