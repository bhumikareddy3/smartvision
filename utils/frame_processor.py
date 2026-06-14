"""
Frame Processor
===============
Orchestrates the full per-frame pipeline:
  1. Resize (optional)
  2. Detect + Track (YOLOv8 + ByteTrack)
  3. Speed estimation
  4. Line crossing check
  5. Zone analytics
  6. Heatmap update
  7. Crowd density update
  8. Alert evaluation
  9. DB logging (batched)
  10. Visualization overlay

Returns the annotated frame + a state dict consumed by the dashboard.
"""

import time
from collections import deque
from typing import Optional, Tuple
import numpy as np
import cv2

from models.detector import ObjectDetector
from trackers.byte_tracker import ObjectTracker
from analytics.counter import LineCounter
from analytics.zone_analytics import ZoneAnalytics
from analytics.heatmap import HeatmapGenerator
from analytics.crowd_density import CrowdDensityMonitor
from analytics.speed_estimator import SpeedEstimator
from analytics.alert_manager import AlertManager
from utils.visualization import (
    draw_detections, draw_trail, draw_line,
    draw_zones, draw_hud,
)
from database.db_manager import DatabaseManager
import config.settings as cfg


class FrameProcessor:
    """
    Centralised pipeline controller.

    Parameters
    ----------
    db          : DatabaseManager instance
    session_id  : Active session ID for DB logging
    frame_size  : (height, width) of the expected video frames
    """

    def __init__(
        self,
        db: DatabaseManager,
        session_id: int,
        frame_size: Tuple[int, int] = (720, 1280),
    ):
        self.db         = db
        self.session_id = session_id
        self.frame_size = frame_size

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.detector = ObjectDetector(
            model_name=cfg.YOLO_MODEL,
            confidence=cfg.DETECTION_CONFIDENCE,
            iou=cfg.DETECTION_IOU,
            target_classes=cfg.TARGET_CLASSES,
            device=cfg.DEVICE,
        )
        self.tracker   = ObjectTracker(history_len=cfg.TRACK_BUFFER)
        self.counter   = LineCounter(
            line_y=frame_size[0] * cfg.DEFAULT_LINE_POSITION,
            frame_width=frame_size[1],
            tolerance=cfg.CROSSING_TOLERANCE,
        )
        self.zones     = ZoneAnalytics()
        self.heatmap   = HeatmapGenerator(
            frame_size=frame_size,
            decay=cfg.HEATMAP_DECAY,
            alpha=cfg.HEATMAP_ALPHA,
        )
        self.density   = CrowdDensityMonitor(
            frame_size=frame_size,
            grid_rows=cfg.DENSITY_GRID_ROWS,
            grid_cols=cfg.DENSITY_GRID_COLS,
        )
        self.speed     = SpeedEstimator(
            pixels_per_meter=cfg.PIXELS_PER_METER,
        )
        self.alerts    = AlertManager(
            occupancy_threshold=cfg.ALERT_OCCUPANCY_THRESHOLD,
            density_threshold=cfg.ALERT_DENSITY_THRESHOLD,
        )

        # ── Runtime state ─────────────────────────────────────────────────────
        self._frame_num    = 0
        self._fps_times: deque = deque(maxlen=30)
        self.show_heatmap  = False
        self.show_trails   = True
        self.show_hud      = True
        self._log_every    = 5      # log to DB every N frames (perf tuning)

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Run the full pipeline on one frame.

        Returns
        -------
        annotated_frame : np.ndarray
        state           : dict with keys used by the Streamlit dashboard
        """
        t0 = time.monotonic()
        self._frame_num += 1
        h, w = frame.shape[:2]

        # ── 0. Optional resize ────────────────────────────────────────────────
        if cfg.RESIZE_WIDTH > 0 and w != cfg.RESIZE_WIDTH:
            scale = cfg.RESIZE_WIDTH / w
            frame = cv2.resize(frame, (cfg.RESIZE_WIDTH, int(h * scale)))
            h, w  = frame.shape[:2]
            self._sync_frame_size((h, w))

        # ── 1. Detect + Track ─────────────────────────────────────────────────
        result     = self.detector.track(frame)
        raw_tracks = self.detector.parse_tracks(result)

        # ── 2. Tracker state update ───────────────────────────────────────────
        tracks = self.tracker.update(raw_tracks)

        # ── 3. Speed estimation ───────────────────────────────────────────────
        tracks = self.speed.update(tracks)

        # ── 4. Line crossing ──────────────────────────────────────────────────
        tracks = self.counter.process_tracks(tracks)

        # ── 5. Zone analytics ─────────────────────────────────────────────────
        zone_events = self.zones.process_tracks(tracks)

        # ── 6. Heatmap + density ──────────────────────────────────────────────
        self.heatmap.update(tracks)
        self.density.update(tracks)

        # ── 7. Alerts ─────────────────────────────────────────────────────────
        new_alerts = self.alerts.process_all(
            occupancy=len(tracks),
            density=self.density.density,
            person_count=self.density.person_count,
            track_list=tracks,
        )

        # ── 8. DB logging (every _log_every frames) ───────────────────────────
        if self._frame_num % self._log_every == 0:
            self._log_to_db(tracks, zone_events, new_alerts)

        # ── 9. Drawing ────────────────────────────────────────────────────────
        vis = frame.copy()
        if self.show_heatmap:
            vis = self.heatmap.overlay(vis)
        draw_zones(vis, self.zones.get_zones())
        if self.show_trails:
            draw_trail(vis, tracks)
        draw_detections(vis, tracks)
        draw_line(vis, int(self.counter.line_y),
                  self.counter.count_in, self.counter.count_out)
        if self.show_hud:
            draw_hud(vis, self.fps, len(tracks),
                     self.tracker.active_count(), self.density.level())

        # ── 10. FPS tracking ──────────────────────────────────────────────────
        self._fps_times.append(time.monotonic() - t0)

        state = self._build_state(tracks, new_alerts)
        return vis, state

    # ── State builder ─────────────────────────────────────────────────────────

    def _build_state(self, tracks: list, new_alerts: list) -> dict:
        class_counts = {}
        for t in tracks:
            n = t["class_name"]
            class_counts[n] = class_counts.get(n, 0) + 1

        return dict(
            frame_number=self._frame_num,
            fps=self.fps,
            total_detected=len(tracks),
            active_tracked=self.tracker.active_count(),
            count_in=self.counter.count_in,
            count_out=self.counter.count_out,
            class_counts=class_counts,
            occupancy_per_zone=self.zones.get_occupancy_summary(),
            crowd_density=self.density.density,
            density_level=self.density.level(),
            person_count=self.density.person_count,
            new_alerts=[a.to_dict() for a in new_alerts],
            recent_alerts=[a.to_dict() for a in self.alerts.recent_alerts],
        )

    # ── DB logging ────────────────────────────────────────────────────────────

    def _log_to_db(self, tracks, zone_events, alerts):
        for t in tracks:
            self.db.log_detection(
                session_id=self.session_id,
                frame_number=self._frame_num,
                track_id=t["track_id"],
                object_type=t["class_name"],
                confidence=t["confidence"],
                bbox=(t["x1"], t["y1"], t["x2"], t["y2"]),
                speed_kmh=t.get("speed_kmh", 0),
            )
            if t.get("crossing"):
                self.db.log_crossing(
                    session_id=self.session_id,
                    track_id=t["track_id"],
                    object_type=t["class_name"],
                    direction=t["crossing"],
                    confidence=t["confidence"],
                )
        for ev in zone_events:
            self.db.log_zone_event(
                session_id=self.session_id,
                zone_id=ev["zone_id"],
                zone_name=ev["zone_name"],
                track_id=ev["track_id"],
                object_type=ev["object_type"],
                event_type=ev["event_type"],
            )
        for a in alerts:
            self.db.log_alert(
                session_id=self.session_id,
                alert_type=a.alert_type,
                message=a.message,
                severity=a.severity,
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sync_frame_size(self, size: Tuple[int, int]):
        self.frame_size = size
        self.heatmap.resize(size)
        self.density.resize(size)
        self.counter.set_line(
            size[0] * cfg.DEFAULT_LINE_POSITION, size[1]
        )

    @property
    def fps(self) -> float:
        if not self._fps_times:
            return 0.0
        avg = sum(self._fps_times) / len(self._fps_times)
        return round(1.0 / avg, 1) if avg > 0 else 0.0

    def reset(self):
        self.detector.reset()
        self.tracker.reset()
        self.counter.reset_counts()
        self.zones.clear_zones()
        self.heatmap.reset()
        self.speed.reset()
        self._frame_num = 0
        self._fps_times.clear()
