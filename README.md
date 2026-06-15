# 👁️ SmartVision — Real-Time Object Tracking & Counting System

A production-quality computer vision system that **detects, tracks, and counts** multiple object classes from webcam, video files, or RTSP streams. Built on YOLOv8 + ByteTrack with a live Streamlit dashboard.

---
## Dashboard Preview

![Dashboard] see in screenshots folder
## 🎥 Demo Video

[Watch Demo](https://drive.google.com/file/d/1dsTr0fS9pN4Ye12riRM_y4mbMKH5bM6E/view?usp=sharing)


## ✨ Features

| Feature | Description |
|---|---|
| **Object Detection** | YOLOv8 pretrained — persons, cars, motorcycles, buses, trucks |
| **Multi-Object Tracking** | ByteTrack with persistent IDs across frames & occlusions |
| **Line Crossing Counter** | Virtual counting line — IN / OUT counts, no duplicates |
| **Zone Analytics** | Custom rectangular ROI zones with live occupancy |
| **Heatmap** | Decaying movement heatmap of frequently traversed areas |
| **Crowd Density** | Grid-based crowd density score + level (Low→Critical) |
| **Speed Estimation** | Per-vehicle speed estimation in km/h |
| **Alert System** | Configurable alerts for occupancy, density, and speed |
| **Data Logging** | SQLite with sessions, detections, crossings, zones, alerts |
| **Reporting** | CSV export, daily text reports, Plotly charts |
| **Live Dashboard** | Streamlit with real-time metrics, charts, and data tables |

---

## 🚀 Quick Start

### 1. Clone / unzip the project

```bash
cd smartvision
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration:** If you have an NVIDIA GPU with CUDA, also run:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser to **http://localhost:8501**

---

## 📁 Project Structure

```
smartvision/
│
├── app.py                    # Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
│
├── config/
│   ├── __init__.py
│   └── settings.py           # All tunable parameters
│
├── database/
│   ├── __init__.py
│   └── db_manager.py         # SQLite manager (thread-safe WAL mode)
│
├── models/
│   ├── __init__.py
│   └── detector.py           # YOLOv8 wrapper + ByteTrack integration
│
├── trackers/
│   ├── __init__.py
│   └── byte_tracker.py       # Track state management & position history
│
├── analytics/
│   ├── __init__.py
│   ├── counter.py            # Virtual counting line (IN/OUT)
│   ├── zone_analytics.py     # Zone-based occupancy monitoring
│   ├── heatmap.py            # Decaying movement heatmap
│   ├── crowd_density.py      # Grid-based crowd density
│   ├── speed_estimator.py    # Vehicle speed estimation
│   └── alert_manager.py      # Threshold-based alert system
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # OpenCV drawing utilities
│   └── frame_processor.py    # Full per-frame pipeline orchestrator
│
├── reports/
│   ├── __init__.py
│   └── report_generator.py   # CSV exports, text reports, Plotly charts
│
├── assets/                   # Static files (icons, sample images)
├── database/                 # SQLite database files (auto-created)
└── reports/                  # Generated report files (auto-created)
```

---

## ⚙️ Configuration

All tunable parameters live in **`config/settings.py`**:

```python
YOLO_MODEL           = "yolov8n.pt"   # n/s/m/l/x — speed vs accuracy trade-off
DETECTION_CONFIDENCE = 0.45
TARGET_CLASSES       = [0, 2, 3, 5, 7]  # person, car, moto, bus, truck
DEFAULT_LINE_POSITION = 0.5            # counting line at 50% frame height
PIXELS_PER_METER     = 15.0           # calibrate for speed estimation
ALERT_OCCUPANCY_THRESHOLD = 20
ALERT_DENSITY_THRESHOLD   = 0.65
```

---

## 🗄️ Database Schema

```sql
-- Recording sessions
sessions (id, source, started_at, ended_at, total_frames)

-- Per-frame object detections
detections (id, session_id, timestamp, frame_number,
            track_id, object_type, confidence,
            x1, y1, x2, y2, speed_kmh)

-- Line-crossing events
crossings (id, session_id, timestamp,
           track_id, object_type, direction, confidence)

-- Zone entry/exit events
zone_events (id, session_id, timestamp,
             zone_id, zone_name, track_id, object_type, event_type)

-- Alert log
alerts (id, session_id, timestamp,
        alert_type, message, severity)
```

---

## 📊 Dashboard Tabs

| Tab | Contents |
|---|---|
| **Live Feed** | Video stream + 6 live metrics + alert banner |
| **Analytics** | Class counts, zone occupancy, 4 Plotly charts |
| **Data & Reports** | Table previews, CSV/report downloads |
| **Alerts** | Alert history + current threshold display |
| **About** | Architecture overview, tech stack, schema docs |

---

## 🎯 Speed Estimation Calibration

Pixel-to-metre mapping depends on camera height, angle, and focal length.
To calibrate:

1. Place a reference object of known length (e.g. a 2 m barrier) in the scene.
2. Measure its pixel width in the frame.
3. Set `PIXELS_PER_METER = pixel_width / real_length_metres` in `settings.py`.

---

## 🏎️ Performance Tips

| Setting | Effect |
|---|---|
| `YOLO_MODEL = "yolov8n.pt"` | Fastest — use for real-time on CPU |
| `RESIZE_WIDTH = 640` | Reduce input resolution to increase FPS |
| `FRAME_SKIP = 1` | Process every other frame |
| CUDA GPU | 3–10× faster — install PyTorch with CUDA |

---

## 📦 Requirements Summary

```
ultralytics   >= 8.0      # YOLOv8 + ByteTrack
opencv-python >= 4.8
streamlit     >= 1.28
plotly        >= 5.17
pandas        >= 2.0
numpy         >= 1.24
scipy         >= 1.11
SQLAlchemy    >= 2.0
psutil        >= 5.9
```

---

## 📝 License

MIT — free to use, modify, and distribute.
