"""
Visualization Utilities
========================
All OpenCV drawing helpers for bounding boxes, labels, lines, and zones.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2


# ── Color palette ─────────────────────────────────────────────────────────────

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 127),    # person  → green
    2: (0, 165, 255),    # car     → orange
    3: (255, 0, 255),    # moto    → magenta
    5: (50, 50, 255),    # bus     → red
    7: (50, 200, 255),   # truck   → yellow-ish
}
DEFAULT_COLOR = (200, 200, 200)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 1


def _color(class_id: int) -> Tuple[int, int, int]:
    return CLASS_COLORS.get(class_id, DEFAULT_COLOR)


# ── Bounding boxes & labels ───────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, track_list: List[dict]) -> np.ndarray:
    """
    Draw bounding boxes, class labels, track IDs, confidence, and speed.
    Works IN-PLACE and also returns the frame.
    """
    for t in track_list:
        x1, y1, x2, y2 = int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"])
        cid    = t["class_id"]
        color  = _color(cid)
        tid    = t["track_id"]
        name   = t["class_name"]
        conf   = t["confidence"]
        speed  = t.get("speed_kmh", 0)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label  = f"{name} #{tid} {conf:.0%}"
        if speed > 1:
            label += f" {speed:.0f}km/h"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA,
        )

        # Centre dot
        cv2.circle(frame, (int(t["cx"]), int(t["cy"])), 3, color, -1)

    return frame


def draw_trail(frame: np.ndarray, track_list: List[dict]) -> np.ndarray:
    """Draw motion trails from track position history."""
    for t in track_list:
        positions = t.get("positions", [])
        color = _color(t["class_id"])
        for i in range(1, len(positions)):
            pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            alpha = i / len(positions)   # fade older points
            trail_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pt1, pt2, trail_color, 1)
    return frame


# ── Counting line ─────────────────────────────────────────────────────────────

def draw_line(
    frame: np.ndarray,
    line_y: int,
    count_in: int,
    count_out: int,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), color, 3)

    # IN / OUT counters
    cv2.putText(frame, f"IN:  {count_in}",  (10, line_y - 10),
                FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"OUT: {count_out}", (10, line_y + 25),
                FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


# ── Zones ────────────────────────────────────────────────────────────────────

def draw_zones(frame: np.ndarray, zones, alpha: float = 0.25) -> np.ndarray:
    """Render semi-transparent zone overlays."""
    overlay = frame.copy()
    for zone in zones:
        x1, y1 = int(zone.x1), int(zone.y1)
        x2, y2 = int(zone.x2), int(zone.y2)
        color  = zone.color

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame,   (x1, y1), (x2, y2), color,  2)

        label = f"{zone.name}: {zone.occupancy}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 6, y1 + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 + th + 3),
                    FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


# ── Heatmap overlay ───────────────────────────────────────────────────────────

def draw_heatmap_overlay(frame: np.ndarray, heatmap_gen) -> np.ndarray:
    """Thin wrapper that calls heatmap_gen.overlay(frame)."""
    return heatmap_gen.overlay(frame)


# ── HUD overlay ───────────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    fps: float,
    total_detected: int,
    active_tracked: int,
    density_level: str,
) -> np.ndarray:
    """Top-left stats HUD."""
    lines = [
        f"FPS:      {fps:.1f}",
        f"Detected: {total_detected}",
        f"Tracked:  {active_tracked}",
        f"Density:  {density_level}",
    ]
    x, y = 10, 24
    for line in lines:
        cv2.putText(frame, line, (x, y), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return frame
