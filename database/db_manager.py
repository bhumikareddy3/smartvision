"""
Database Manager
================
Handles all SQLite persistence for SmartVision.

Tables:
  detections  — one row per detected object per frame
  crossings   — line-crossing events (IN / OUT)
  alerts      — alert log
  sessions    — video session metadata
"""

import sqlite3
import threading
from datetime import datetime, date
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

import pandas as pd


class DatabaseManager:
    """Thread-safe SQLite manager with connection pooling via threading.local."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_schema()

    # ── Connection management ─────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection, creating it if needed."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        """Create tables if they do not already exist."""
        with self._cursor() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    source      TEXT NOT NULL,
                    started_at  TEXT NOT NULL,
                    ended_at    TEXT,
                    total_frames INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      INTEGER REFERENCES sessions(id),
                    timestamp       TEXT NOT NULL,
                    frame_number    INTEGER NOT NULL,
                    track_id        INTEGER NOT NULL,
                    object_type     TEXT NOT NULL,
                    confidence      REAL NOT NULL,
                    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                    speed_kmh       REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS crossings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER REFERENCES sessions(id),
                    timestamp   TEXT NOT NULL,
                    track_id    INTEGER NOT NULL,
                    object_type TEXT NOT NULL,
                    direction   TEXT NOT NULL CHECK(direction IN ('IN','OUT')),
                    confidence  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS zone_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER REFERENCES sessions(id),
                    timestamp   TEXT NOT NULL,
                    zone_id     INTEGER NOT NULL,
                    zone_name   TEXT NOT NULL,
                    track_id    INTEGER NOT NULL,
                    object_type TEXT NOT NULL,
                    event_type  TEXT NOT NULL CHECK(event_type IN ('ENTER','EXIT'))
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER REFERENCES sessions(id),
                    timestamp   TEXT NOT NULL,
                    alert_type  TEXT NOT NULL,
                    message     TEXT NOT NULL,
                    severity    TEXT NOT NULL DEFAULT 'WARNING'
                );

                CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id);
                CREATE INDEX IF NOT EXISTS idx_det_ts      ON detections(timestamp);
                CREATE INDEX IF NOT EXISTS idx_cross_sess  ON crossings(session_id);
                CREATE INDEX IF NOT EXISTS idx_cross_ts    ON crossings(timestamp);
            """)

    # ── Session CRUD ──────────────────────────────────────────────────────────

    def start_session(self, source: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (source, started_at) VALUES (?,?)",
                (source, datetime.now().isoformat()),
            )
            return cur.lastrowid

    def end_session(self, session_id: int, total_frames: int):
        with self._cursor() as cur:
            cur.execute(
                "UPDATE sessions SET ended_at=?, total_frames=? WHERE id=?",
                (datetime.now().isoformat(), total_frames, session_id),
            )

    # ── Insert helpers ────────────────────────────────────────────────────────

    def log_detection(
        self,
        session_id: int,
        frame_number: int,
        track_id: int,
        object_type: str,
        confidence: float,
        bbox: tuple,
        speed_kmh: float = 0.0,
    ):
        x1, y1, x2, y2 = bbox
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO detections
                   (session_id, timestamp, frame_number, track_id,
                    object_type, confidence, x1, y1, x2, y2, speed_kmh)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    datetime.now().isoformat(),
                    frame_number,
                    track_id,
                    object_type,
                    round(confidence, 4),
                    x1, y1, x2, y2,
                    round(speed_kmh, 2),
                ),
            )

    def log_crossing(
        self,
        session_id: int,
        track_id: int,
        object_type: str,
        direction: str,
        confidence: float,
    ):
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO crossings
                   (session_id, timestamp, track_id, object_type, direction, confidence)
                   VALUES (?,?,?,?,?,?)""",
                (
                    session_id,
                    datetime.now().isoformat(),
                    track_id,
                    object_type,
                    direction,
                    round(confidence, 4),
                ),
            )

    def log_zone_event(
        self,
        session_id: int,
        zone_id: int,
        zone_name: str,
        track_id: int,
        object_type: str,
        event_type: str,
    ):
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO zone_events
                   (session_id, timestamp, zone_id, zone_name,
                    track_id, object_type, event_type)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    session_id,
                    datetime.now().isoformat(),
                    zone_id,
                    zone_name,
                    track_id,
                    object_type,
                    event_type,
                ),
            )

    def log_alert(
        self,
        session_id: int,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
    ):
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO alerts
                   (session_id, timestamp, alert_type, message, severity)
                   VALUES (?,?,?,?,?)""",
                (
                    session_id,
                    datetime.now().isoformat(),
                    alert_type,
                    message,
                    severity,
                ),
            )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_crossings_df(self, session_id: Optional[int] = None) -> pd.DataFrame:
        query = "SELECT * FROM crossings"
        params = []
        if session_id:
            query += " WHERE session_id=?"
            params.append(session_id)
        query += " ORDER BY timestamp DESC"
        conn = self._get_conn()
        return pd.read_sql_query(query, conn, params=params)

    def get_detections_df(self, session_id: Optional[int] = None) -> pd.DataFrame:
        query = "SELECT * FROM detections"
        params = []
        if session_id:
            query += " WHERE session_id=?"
            params.append(session_id)
        query += " ORDER BY timestamp DESC LIMIT 5000"
        conn = self._get_conn()
        return pd.read_sql_query(query, conn, params=params)

    def get_alerts_df(self, session_id: Optional[int] = None) -> pd.DataFrame:
        query = "SELECT * FROM alerts"
        params = []
        if session_id:
            query += " WHERE session_id=?"
            params.append(session_id)
        query += " ORDER BY timestamp DESC"
        conn = self._get_conn()
        return pd.read_sql_query(query, conn, params=params)

    def get_daily_summary(self, target_date: Optional[date] = None) -> pd.DataFrame:
        """Return per-class crossing counts for a given date (default: today)."""
        d = target_date or date.today()
        query = """
            SELECT object_type,
                   SUM(CASE WHEN direction='IN'  THEN 1 ELSE 0 END) AS count_in,
                   SUM(CASE WHEN direction='OUT' THEN 1 ELSE 0 END) AS count_out,
                   COUNT(*) AS total
              FROM crossings
             WHERE DATE(timestamp) = ?
             GROUP BY object_type
        """
        conn = self._get_conn()
        return pd.read_sql_query(query, conn, params=[str(d)])

    def get_hourly_counts(self, session_id: Optional[int] = None) -> pd.DataFrame:
        """Hourly crossing counts — used for time-series charts."""
        where = "WHERE session_id=?" if session_id else ""
        params = [session_id] if session_id else []
        query = f"""
            SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) AS hour,
                   direction,
                   COUNT(*) AS count
              FROM crossings
              {where}
             GROUP BY hour, direction
             ORDER BY hour
        """
        conn = self._get_conn()
        return pd.read_sql_query(query, conn, params=params)

    def export_to_csv(self, output_path: str, session_id: Optional[int] = None):
        """Export crossings + detections to a CSV file."""
        df = self.get_crossings_df(session_id)
        df.to_csv(output_path, index=False)
        return output_path
