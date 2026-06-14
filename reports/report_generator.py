"""
Report Generator
================
Generates:
  • CSV exports (crossings, detections)
  • Daily summary text report
  • Plotly figures for embedding in the dashboard

All output goes to the reports/ directory.
"""

import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from database.db_manager import DatabaseManager
import config.settings as cfg


REPORTS_DIR = cfg.REPORTS_DIR


class ReportGenerator:
    def __init__(self, db: DatabaseManager):
        self.db = db

    # ── CSV Export ────────────────────────────────────────────────────────────

    def export_crossings_csv(
        self, session_id: Optional[int] = None, filename: Optional[str] = None
    ) -> str:
        df = self.db.get_crossings_df(session_id)
        fname = filename or f"crossings_{date.today()}.csv"
        path  = REPORTS_DIR / fname
        df.to_csv(path, index=False)
        return str(path)

    def export_detections_csv(
        self, session_id: Optional[int] = None, filename: Optional[str] = None
    ) -> str:
        df = self.db.get_detections_df(session_id)
        fname = filename or f"detections_{date.today()}.csv"
        path  = REPORTS_DIR / fname
        df.to_csv(path, index=False)
        return str(path)

    # ── Daily text report ─────────────────────────────────────────────────────

    def daily_report_text(self, target_date: Optional[date] = None) -> str:
        d   = target_date or date.today()
        df  = self.db.get_daily_summary(d)
        adf = self.db.get_alerts_df()

        lines = [
            f"SmartVision Daily Report — {d}",
            "=" * 50,
            "",
            "CROSSING SUMMARY",
            "-" * 30,
        ]

        if df.empty:
            lines.append("  No crossings recorded.")
        else:
            total_in  = df["count_in"].sum()
            total_out = df["count_out"].sum()
            lines.append(f"  Total IN:  {total_in}")
            lines.append(f"  Total OUT: {total_out}")
            lines.append("")
            lines.append("  By object type:")
            for _, row in df.iterrows():
                lines.append(
                    f"    {row['object_type']:12s}  IN={row['count_in']:4d}  OUT={row['count_out']:4d}"
                )

        lines += [
            "",
            "ALERTS",
            "-" * 30,
        ]
        today_alerts = adf[adf["timestamp"].str.startswith(str(d))] if not adf.empty else adf
        if today_alerts.empty:
            lines.append("  No alerts recorded.")
        else:
            for _, row in today_alerts.iterrows():
                lines.append(f"  [{row['severity']}] {row['timestamp'][:19]}  {row['message']}")

        lines += ["", f"Report generated: {datetime.now().isoformat()[:19]}"]
        return "\n".join(lines)

    # ── Plotly charts ─────────────────────────────────────────────────────────

    def fig_hourly_crossings(self, session_id: Optional[int] = None) -> go.Figure:
        """Line chart of IN/OUT counts per hour."""
        df = self.db.get_hourly_counts(session_id)
        if df.empty:
            return self._empty_fig("No crossing data yet")

        fig = px.line(
            df, x="hour", y="count", color="direction",
            title="Hourly Crossing Counts",
            color_discrete_map={"IN": "#00d084", "OUT": "#ff4b4b"},
            labels={"hour": "Time", "count": "Crossings"},
        )
        fig.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#fafafa",
            legend_title="Direction",
        )
        return fig

    def fig_class_distribution(self, session_id: Optional[int] = None) -> go.Figure:
        """Pie chart of detected object classes."""
        df = self.db.get_crossings_df(session_id)
        if df.empty:
            return self._empty_fig("No crossing data yet")

        counts = df["object_type"].value_counts().reset_index()
        counts.columns = ["object_type", "count"]
        fig = px.pie(
            counts, names="object_type", values="count",
            title="Object Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#fafafa",
        )
        return fig

    def fig_in_out_bar(self, session_id: Optional[int] = None) -> go.Figure:
        """Grouped bar: IN vs OUT per object class."""
        df = self.db.get_daily_summary()
        if df.empty:
            return self._empty_fig("No crossing data yet")

        fig = go.Figure(data=[
            go.Bar(name="IN",  x=df["object_type"], y=df["count_in"],
                   marker_color="#00d084"),
            go.Bar(name="OUT", x=df["object_type"], y=df["count_out"],
                   marker_color="#ff4b4b"),
        ])
        fig.update_layout(
            barmode="group",
            title="Today's IN / OUT by Object Type",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#fafafa",
        )
        return fig

    def fig_speed_distribution(self, session_id: Optional[int] = None) -> go.Figure:
        """Histogram of recorded vehicle speeds."""
        df = self.db.get_detections_df(session_id)
        df = df[df["speed_kmh"] > 1]   # filter stopped objects
        if df.empty:
            return self._empty_fig("No speed data yet")

        fig = px.histogram(
            df, x="speed_kmh", nbins=30,
            title="Vehicle Speed Distribution (km/h)",
            color_discrete_sequence=["#f5a623"],
            labels={"speed_kmh": "Speed (km/h)"},
        )
        fig.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#fafafa",
        )
        return fig

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_fig(message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#888"),
        )
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa"
        )
        return fig
