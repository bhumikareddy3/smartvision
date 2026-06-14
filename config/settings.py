"""
SmartVision Configuration Settings
===================================
Central configuration for all system parameters.
Modify these values to tune detection, tracking, and dashboard behavior.
"""

import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for d in [DATABASE_DIR, REPORTS_DIR, ASSETS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_PATH = DATABASE_DIR / "smartvision.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# ── YOLOv8 Detection ──────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"          # yolov8n/s/m/l/x — n=fastest, x=most accurate
DETECTION_CONFIDENCE = 0.45        # Minimum confidence threshold (0–1)
DETECTION_IOU = 0.45               # IoU threshold for NMS
DEVICE = "auto"                    # "auto", "cpu", "cuda", "mps"

# Target classes from COCO dataset
# 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = [0, 2, 3, 5, 7]
CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Bounding box colors per class (BGR)
CLASS_COLORS = {
    0: (0, 255, 127),    # person  → green
    2: (0, 165, 255),    # car     → orange
    3: (255, 0, 255),    # moto    → magenta
    5: (255, 50, 50),    # bus     → blue-ish red
    7: (50, 50, 255),    # truck   → red
}

# ── ByteTrack Tracker ─────────────────────────────────────────────────────────
TRACKER_CONFIG = "bytetrack.yaml"  # Built-in ultralytics tracker config
TRACK_BUFFER = 30                  # Frames to keep lost tracks alive
MATCH_THRESH = 0.8                 # Matching threshold

# ── Counting Line ─────────────────────────────────────────────────────────────
DEFAULT_LINE_POSITION = 0.5        # Fraction of frame height (0=top, 1=bottom)
LINE_COLOR = (0, 0, 255)           # Red counting line
LINE_THICKNESS = 3
CROSSING_TOLERANCE = 5             # Pixels of tolerance for line crossing

# ── Zone Analytics ────────────────────────────────────────────────────────────
MAX_ZONES = 6
ZONE_ALPHA = 0.25                  # Zone overlay transparency

# ── Heatmap ───────────────────────────────────────────────────────────────────
HEATMAP_DECAY = 0.98               # Decay factor per frame (lower = faster fade)
HEATMAP_ALPHA = 0.5                # Blend weight with original frame

# ── Crowd Density ─────────────────────────────────────────────────────────────
DENSITY_GRID_ROWS = 8
DENSITY_GRID_COLS = 8
HIGH_DENSITY_THRESHOLD = 0.6       # Fraction of cells occupied

# ── Speed Estimation ─────────────────────────────────────────────────────────
PIXELS_PER_METER = 15.0            # Calibration: pixels = 1 metre (tune per scene)
FPS_SPEED_WINDOW = 5               # Frames for rolling speed average

# ── Alerts ────────────────────────────────────────────────────────────────────
ALERT_OCCUPANCY_THRESHOLD = 20     # Objects in scene before alert
ALERT_DENSITY_THRESHOLD = 0.65     # Crowd density fraction before alert
ALERT_COOLDOWN_SECONDS = 30        # Seconds between repeated alerts

# ── Performance ───────────────────────────────────────────────────────────────
FRAME_SKIP = 0                     # Skip N frames between detections (0 = every frame)
MAX_FPS = 30                       # Cap display FPS
RESIZE_WIDTH = 960                 # Resize input frame width (0 = no resize)

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_MS = 100         # Streamlit refresh interval
CHART_HISTORY_MINUTES = 60         # How many minutes of history to show on charts
