"""
SmartVision — Cyberpunk Dark Dashboard
=======================================
Dark neon theme applied only to custom HTML components.
All Streamlit native widgets left untouched — sidebar works normally.
No emojis.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import pandas as pd
import streamlit as st

from database.db_manager import DatabaseManager
from utils.frame_processor import FrameProcessor
from analytics.zone_analytics import Zone
from reports.report_generator import ReportGenerator
import config.settings as cfg

st.set_page_config(
    page_title="SmartVision",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Share+Tech+Mono&display=swap');

/* ── Page background ── */
.stApp { background-color: #020913; }
.block-container {
    padding-top: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── Page title ── */
.sv-title {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    color: #00f5ff;
    text-shadow: 0 0 30px rgba(0,245,255,0.7), 0 0 60px rgba(0,245,255,0.2);
    letter-spacing: 0.12em;
    margin: 0;
    line-height: 1.2;
    overflow: visible;
}
.sv-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #ff00cc;
    letter-spacing: 0.3em;
    text-shadow: 0 0 10px rgba(255,0,204,0.5);
    margin-top: 6px;
    display: block;
}
.sv-header {
    border-bottom: 1px solid #0d3354;
    padding-bottom: 14px;
    margin-bottom: 20px;
    margin-top: 10px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.sv-header-right {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: #2e4f65;
    letter-spacing: 0.18em;
    text-align: right;
    line-height: 2;
}

/* ── Status pill ── */
.sv-pill-live {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #00ffaa;
    background: rgba(0,255,170,0.07);
    border: 1px solid rgba(0,255,170,0.3);
    padding: 3px 12px;
    letter-spacing: 0.18em;
    text-shadow: 0 0 8px rgba(0,255,170,0.6);
}
.sv-pill-off {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #2e4f65;
    background: transparent;
    border: 1px solid #0d3354;
    padding: 3px 12px;
    letter-spacing: 0.18em;
}

/* ── Metric cards ── */
.metric-box {
    background: #060e1c;
    border: 1px solid #0d3354;
    border-top: 2px solid #00f5ff;
    padding: 18px 12px 14px;
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00f5ff;
    text-shadow: 0 0 16px rgba(0,245,255,0.6);
    margin: 0;
    display: block;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: #2e6a8a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 6px;
    display: block;
}
.mv-green  { color: #00ffaa; text-shadow: 0 0 14px rgba(0,255,170,0.6); }
.mv-red    { color: #ff2255; text-shadow: 0 0 14px rgba(255,34,85,0.6); }
.mv-yellow { color: #ffe600; text-shadow: 0 0 14px rgba(255,230,0,0.6); }
.mv-mag    { color: #ff00cc; text-shadow: 0 0 14px rgba(255,0,204,0.6); }
.mt-green  { border-top-color: #00ffaa; }
.mt-red    { border-top-color: #ff2255; }
.mt-yellow { border-top-color: #ffe600; }
.mt-mag    { border-top-color: #ff00cc; }

/* ── Feed chrome ── */
.sv-feed-bar {
    background: #060e1c;
    border: 1px solid #0d3354;
    border-bottom: none;
    padding: 6px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #00f5ff;
    letter-spacing: 0.2em;
    display: flex;
    justify-content: space-between;
}
.sv-feed-bar span { color: #2e4f65; }
.sv-feed-foot {
    background: #060e1c;
    border: 1px solid #0d3354;
    border-top: none;
    padding: 4px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.52rem;
    color: #2e4f65;
    letter-spacing: 0.14em;
    display: flex;
    justify-content: space-between;
}

/* ── Section headers ── */
.sv-section {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #00f5ff;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    border-left: 2px solid #00f5ff;
    padding-left: 9px;
    margin: 4px 0 12px;
    text-shadow: 0 0 8px rgba(0,245,255,0.4);
    display: block;
}

/* ── Alert boxes ── */
.alert-box {
    background: #1a0610;
    border-left: 3px solid #ff2255;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #ff6680;
    letter-spacing: 0.06em;
}
.warn-box {
    background: #1a1500;
    border-left: 3px solid #ffe600;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #ffe980;
    letter-spacing: 0.06em;
}

/* ── Sidebar title ── */
.sv-sidebar-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.05rem;
    font-weight: 900;
    color: #00f5ff;
    text-shadow: 0 0 16px rgba(0,245,255,0.6);
    letter-spacing: 0.14em;
    display: block;
    margin-bottom: 2px;
}
.sv-sidebar-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.55rem;
    color: #2e6a8a;
    letter-spacing: 0.2em;
    display: block;
    margin-bottom: 8px;
}

/* ── Empty state ── */
.sv-empty {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #2e4f65;
    text-align: center;
    border: 1px dashed #0d3354;
    padding: 40px 20px;
    letter-spacing: 0.2em;
    line-height: 2.2;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in dict(running=False, processor=None, db=None,
                 session_id=None, reporter=None,
                 alerts=[], stats={}, video_path=None).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <span class="sv-sidebar-title">SMARTVISION</span>
    <span class="sv-sidebar-sub">REAL-TIME NEURAL TRACKER // v2</span>
    """, unsafe_allow_html=True)
    st.divider()

    if st.session_state["running"]:
        st.success("SYSTEM ACTIVE")
    else:
        st.info("STANDBY")

    st.divider()

    # ── Video Source ──────────────────────────────────────────────────────────
    st.markdown('<span class="sv-section">Input Source</span>', unsafe_allow_html=True)
    source_type = st.selectbox("Input Type", ["Webcam", "Video File", "RTSP Stream"])

    video_source = None
    if source_type == "Webcam":
        cam_idx = st.number_input("Camera Index", min_value=0, max_value=10, value=0)
        video_source = int(cam_idx)
    elif source_type == "Video File":
        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.flush()
            video_source = tmp.name
            st.session_state["video_path"] = video_source
        elif st.session_state.get("video_path"):
            video_source = st.session_state["video_path"]
    else:
        video_source = st.text_input("RTSP URL", "rtsp://camera/stream")

    st.divider()

    # ── Detection ─────────────────────────────────────────────────────────────
    st.markdown('<span class="sv-section">Detection Settings</span>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, cfg.DETECTION_CONFIDENCE, 0.05)
    st.selectbox("YOLOv8 Model", ["yolov8n.pt  (Fastest)", "yolov8s.pt", "yolov8m.pt  (Accurate)"])

    st.caption("Track these classes:")
    c1, c2 = st.columns(2)
    with c1:
        tp = st.checkbox("Person",  value=True)
        tc = st.checkbox("Car",     value=True)
        tb = st.checkbox("Bus",     value=True)
    with c2:
        tm = st.checkbox("Moto",    value=True)
        tt = st.checkbox("Truck",   value=True)

    target_classes = (
        ([0] if tp else []) + ([2] if tc else []) +
        ([3] if tm else []) + ([5] if tb else []) + ([7] if tt else [])
    )

    st.divider()

    # ── Counting Line ─────────────────────────────────────────────────────────
    st.markdown('<span class="sv-section">Counting Line</span>', unsafe_allow_html=True)
    line_pos = st.slider("Position (% from top)", 10, 90,
                         int(cfg.DEFAULT_LINE_POSITION * 100), 5)

    st.divider()

    # ── Display ───────────────────────────────────────────────────────────────
    st.markdown('<span class="sv-section">Display Options</span>', unsafe_allow_html=True)
    show_heatmap = st.toggle("Heatmap Overlay",  value=False)
    show_trails  = st.toggle("Motion Trails",    value=True)
    show_hud     = st.toggle("Stats HUD",        value=True)

    st.divider()

    # ── Alerts ────────────────────────────────────────────────────────────────
    st.markdown('<span class="sv-section">Alert Thresholds</span>', unsafe_allow_html=True)
    occ_threshold  = st.number_input("Max Occupancy",      10, 200, cfg.ALERT_OCCUPANCY_THRESHOLD)
    dens_threshold = st.slider("Max Crowd Density (%)", 10, 100,
                               int(cfg.ALERT_DENSITY_THRESHOLD * 100))

    st.divider()

    # ── Zones ─────────────────────────────────────────────────────────────────
    st.markdown('<span class="sv-section">Monitor Zones</span>', unsafe_allow_html=True)
    n_zones   = st.number_input("Number of Zones", 0, cfg.MAX_ZONES, 0)
    zone_defs = []
    for i in range(int(n_zones)):
        with st.expander(f"Zone {i+1}"):
            zname = st.text_input("Name",  value=f"Zone {i+1}", key=f"zn_{i}")
            zx1   = st.number_input("X1",  0, 3840, 0,   key=f"zx1_{i}")
            zy1   = st.number_input("Y1",  0, 2160, 0,   key=f"zy1_{i}")
            zx2   = st.number_input("X2",  0, 3840, 640, key=f"zx2_{i}")
            zy2   = st.number_input("Y2",  0, 2160, 480, key=f"zy2_{i}")
            zone_defs.append(dict(id=i, name=zname, x1=zx1, y1=zy1, x2=zx2, y2=zy2))

    st.divider()

    # ── Start / Stop ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start", type="primary", use_container_width=True)
    with col2:
        stop_btn  = st.button("Stop",  type="secondary", use_container_width=True)

    if start_btn and video_source is not None:
        db        = DatabaseManager(cfg.DATABASE_PATH)
        sid       = db.start_session(str(video_source))
        processor = FrameProcessor(db, sid)

        processor.detector.confidence        = conf_thresh
        processor.detector.target_classes    = target_classes or cfg.TARGET_CLASSES
        processor.show_heatmap               = show_heatmap
        processor.show_trails                = show_trails
        processor.show_hud                   = show_hud
        processor.alerts.occupancy_threshold = occ_threshold
        processor.alerts.density_threshold   = dens_threshold / 100

        processor.zones.clear_zones()
        for zd in zone_defs:
            processor.zones.add_zone(Zone(**zd))

        st.session_state.update(
            running=True, processor=processor, db=db,
            session_id=sid, reporter=ReportGenerator(db), alerts=[],
        )

    if stop_btn:
        if st.session_state.get("db") and st.session_state.get("session_id"):
            proc = st.session_state.get("processor")
            st.session_state["db"].end_session(
                st.session_state["session_id"],
                proc._frame_num if proc else 0,
            )
        st.session_state["running"] = False


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PAGE HEADER
# ═════════════════════════════════════════════════════════════════════════════
is_running = st.session_state["running"]
pill = ('<span class="sv-pill-live">LIVE</span>'
        if is_running else
        '<span class="sv-pill-off">STANDBY</span>')

st.markdown(f"""
<div class="sv-header">
  <div>
    <div class="sv-title">SMARTVISION</div>
    <span class="sv-subtitle">// REAL-TIME OBJECT TRACKING AND COUNTING SYSTEM</span>
  </div>
  <div class="sv-header-right">
    {pill}<br>
    YOLOv8 + BYTETRACK<br>
    SQLite // OPENCV
  </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Live Feed", "Analytics", "Data and Reports", "Alerts", "About"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE FEED
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    if not is_running or video_source is None:
        st.markdown("""
        <div class="sv-empty">
          NO SIGNAL<br>
          <span style="font-size:0.6rem">
            CONFIGURE INPUT SOURCE IN SIDEBAR<br>
            THEN PRESS START TO BEGIN TRACKING
          </span>
        </div>""", unsafe_allow_html=True)
    else:
        # Metric row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        det_ph  = m1.empty()
        trk_ph  = m2.empty()
        in_ph   = m3.empty()
        out_ph  = m4.empty()
        fps_ph  = m5.empty()
        dens_ph = m6.empty()

        alert_ph = st.empty()

        # Feed chrome
        st.markdown("""
        <div class="sv-feed-bar">
          LIVE VIDEO FEED — DETECTION ACTIVE
          <span>BYTETRACK // YOLOv8 // OPENCV</span>
        </div>""", unsafe_allow_html=True)

        frame_ph = st.empty()

        st.markdown("""
        <div class="sv-feed-foot">
          <span>SMARTVISION</span>
          <span>FRAME BUFFER ACTIVE</span>
          <span>SQLite WAL LOGGING</span>
        </div>""", unsafe_allow_html=True)

        processor = st.session_state["processor"]
        processor.show_heatmap = show_heatmap
        processor.show_trails  = show_trails
        processor.show_hud     = show_hud
        processor.counter.set_line(
            processor.frame_size[0] * (line_pos / 100),
            processor.frame_size[1],
        )

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error(f"Cannot open video source: {video_source}")
            st.session_state["running"] = False
        else:
            actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            processor.speed.update_fps(actual_fps)

            try:
                while st.session_state["running"]:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    vis, state = processor.process(frame)
                    st.session_state["stats"] = state

                    # Metric cards
                    det_ph.markdown(
                        f'<div class="metric-box">'
                        f'<span class="metric-value">{state["total_detected"]}</span>'
                        f'<span class="metric-label">Detected</span></div>',
                        unsafe_allow_html=True)
                    trk_ph.markdown(
                        f'<div class="metric-box">'
                        f'<span class="metric-value">{state["active_tracked"]}</span>'
                        f'<span class="metric-label">Tracked</span></div>',
                        unsafe_allow_html=True)
                    in_ph.markdown(
                        f'<div class="metric-box mt-green">'
                        f'<span class="metric-value mv-green">{state["count_in"]}</span>'
                        f'<span class="metric-label">Entry Count</span></div>',
                        unsafe_allow_html=True)
                    out_ph.markdown(
                        f'<div class="metric-box mt-red">'
                        f'<span class="metric-value mv-red">{state["count_out"]}</span>'
                        f'<span class="metric-label">Exit Count</span></div>',
                        unsafe_allow_html=True)
                    fps_ph.markdown(
                        f'<div class="metric-box mt-yellow">'
                        f'<span class="metric-value mv-yellow">{state["fps"]:.0f}</span>'
                        f'<span class="metric-label">FPS</span></div>',
                        unsafe_allow_html=True)
                    dens_ph.markdown(
                        f'<div class="metric-box mt-mag">'
                        f'<span class="metric-value mv-mag">{state["density_level"]}</span>'
                        f'<span class="metric-label">Crowd Density</span></div>',
                        unsafe_allow_html=True)

                    # Video frame
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    frame_ph.image(vis_rgb, use_container_width=True)

                    # Alert banner
                    if state["new_alerts"]:
                        for a in state["new_alerts"]:
                            st.session_state["alerts"].insert(0, a)
                        msgs = " | ".join(a["message"] for a in state["new_alerts"])
                        alert_ph.error(msgs)

            except Exception as e:
                st.error(f"Pipeline error: {e}")
            finally:
                cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    reporter = st.session_state.get("reporter")
    sid      = st.session_state.get("session_id")

    if not reporter:
        st.markdown('<div class="sv-empty">NO SESSION ACTIVE — START A SESSION TO VIEW ANALYTICS</div>', unsafe_allow_html=True)
    else:
        stats = st.session_state.get("stats", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<span class="sv-section">Object Classes — Current Frame</span>', unsafe_allow_html=True)
            cc = stats.get("class_counts", {})
            if cc:
                st.dataframe(
                    pd.DataFrame(cc.items(), columns=["Object", "Count"]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("No objects detected yet.")

        with col2:
            st.markdown('<span class="sv-section">Zone Occupancy</span>', unsafe_allow_html=True)
            occ = stats.get("occupancy_per_zone", {})
            if occ:
                st.dataframe(
                    pd.DataFrame(occ.items(), columns=["Zone", "Occupancy"]),
                    use_container_width=True, hide_index=True,
                )
                d_pct = stats.get("crowd_density", 0)
                d_lvl = stats.get("density_level", "—")
                st.caption(f"Crowd Density: {d_lvl} ({d_pct*100:.1f}%)")
                st.progress(d_pct)
            else:
                st.caption("No zones configured.")

        st.divider()
        st.markdown('<span class="sv-section">Historical Charts</span>', unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(reporter.fig_hourly_crossings(sid),   use_container_width=True, key="chart_hourly")
        with cb:
            st.plotly_chart(reporter.fig_class_distribution(sid), use_container_width=True, key="chart_class")

        cc2, cd = st.columns(2)
        with cc2:
            st.plotly_chart(reporter.fig_in_out_bar(sid),         use_container_width=True, key="chart_inout")
        with cd:
            st.plotly_chart(reporter.fig_speed_distribution(sid), use_container_width=True, key="chart_speed")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DATA AND REPORTS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    reporter = st.session_state.get("reporter")
    sid      = st.session_state.get("session_id")
    db       = st.session_state.get("db")

    if not reporter:
        st.markdown('<div class="sv-empty">NO SESSION ACTIVE — START A SESSION TO ACCESS DATA</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="sv-section">Export Data</span>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)

        with e1:
            if st.button("Export Crossings CSV"):
                path = reporter.export_crossings_csv(sid)
                df   = pd.read_csv(path)
                st.download_button("Download Crossings", df.to_csv(index=False),
                                   "crossings.csv", "text/csv")
        with e2:
            if st.button("Export Detections CSV"):
                path = reporter.export_detections_csv(sid)
                df   = pd.read_csv(path)
                st.download_button("Download Detections", df.to_csv(index=False),
                                   "detections.csv", "text/csv")
        with e3:
            if st.button("Generate Daily Report"):
                txt = reporter.daily_report_text()
                st.download_button("Download Report", txt,
                                   "daily_report.txt", "text/plain")

        st.divider()
        st.markdown('<span class="sv-section">Data Preview</span>', unsafe_allow_html=True)
        table = st.selectbox("Table", ["Crossings", "Detections", "Alerts"])
        df    = (db.get_crossings_df(sid)  if table == "Crossings" else
                 db.get_detections_df(sid) if table == "Detections" else
                 db.get_alerts_df(sid))

        if df.empty:
            st.caption("No records yet.")
        else:
            st.dataframe(df.head(200), use_container_width=True)

        st.divider()
        st.markdown('<span class="sv-section">Today\'s Summary</span>', unsafe_allow_html=True)
        daily = db.get_daily_summary()
        if daily.empty:
            st.caption("No crossings recorded today.")
        else:
            st.dataframe(daily, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ALERTS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<span class="sv-section">Alert Log</span>', unsafe_allow_html=True)
    alerts = st.session_state.get("alerts", [])

    if not alerts:
        st.markdown("""
        <div class="sv-empty">
          NO ALERTS FIRED<br>
          <span style="font-size:0.6rem">SYSTEM NOMINAL — ALERTS APPEAR HERE WHEN THRESHOLDS ARE EXCEEDED</span>
        </div>""", unsafe_allow_html=True)
    else:
        for a in alerts[:50]:
            sev = a.get("severity", "WARNING")
            ts  = a.get("timestamp", "")[:19]
            msg = a.get("message", "")
            if sev == "CRITICAL":
                st.markdown(f'<div class="alert-box">[CRITICAL] {ts} — {msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warn-box">[WARNING] {ts} — {msg}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="sv-section">Current Thresholds</span>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.metric("Max Occupancy",  occ_threshold)
    t2.metric("Max Density",    f"{dens_threshold}%")
    t3.metric("Total Alerts",   len(alerts))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<span class="sv-section">About SmartVision</span>', unsafe_allow_html=True)
    st.markdown("""
SmartVision is a production-quality real-time computer vision system for detecting,
tracking, and counting multiple object types from webcam, video files, or RTSP streams.

---

### Architecture
```
app.py  ->  FrameProcessor  ->  ObjectDetector  (YOLOv8 + ByteTrack)
                            ->  ObjectTracker   (ID state and history)
                            ->  LineCounter     (IN / OUT counting)
                            ->  ZoneAnalytics   (ROI occupancy)
                            ->  HeatmapGenerator
                            ->  CrowdDensityMonitor
                            ->  SpeedEstimator
                            ->  AlertManager
                            ->  DatabaseManager (SQLite)
            ->  ReportGenerator  ->  CSV / Charts / Reports
```

---

### Tech Stack
| Component | Library |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Multi-Object Tracking | ByteTrack |
| Computer Vision | OpenCV 4.8+ |
| Dashboard | Streamlit 1.28+ |
| Charts | Plotly |
| Data | Pandas / NumPy |
| Storage | SQLite (WAL mode) |

---

### Tracked Object Classes
| ID | Class |
|---|---|
| 0 | Person |
| 2 | Car |
| 3 | Motorcycle |
| 5 | Bus |
| 7 | Truck |

---

### Quick Start
```bash
cd smartvision
pip install -r requirements.txt
streamlit run app.py
```
""")