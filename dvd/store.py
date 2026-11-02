"""SQLite persistence: raw AIS, feed gaps, per-pass summaries and per-detection results.

SQLite keeps the project dependency-free; the same tables map directly onto DuckDB or PostgreSQL.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS ais (mmsi INTEGER, time REAL, lon REAL, lat REAL, sog REAL, cog REAL);
CREATE INDEX IF NOT EXISTS ais_time ON ais(time);
CREATE TABLE IF NOT EXISTS ais_gaps (start REAL, end REAL);
CREATE TABLE IF NOT EXISTS passes (
  pass_id TEXT PRIMARY KEY, time REAL, n_det INTEGER, n_matched INTEGER, n_candidates INTEGER,
  n_low_conf INTEGER, n_filtered INTEGER, match_rate REAL, mean_score REAL, ais_msgs INTEGER,
  ais_gap_s REAL, sea_state REAL);
CREATE TABLE IF NOT EXISTS detections (
  pass_id TEXT, det_id INTEGER, lon REAL, lat REAL, length_m REAL, score REAL, shore_km REAL,
  status TEXT, reason TEXT, confidence TEXT, crop TEXT);
"""


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def add_ais(con, df: pd.DataFrame) -> None:
    if not df.empty:
        df[["mmsi", "time", "lon", "lat", "sog", "cog"]].to_sql("ais", con, if_exists="append", index=False)
        con.commit()


def add_gap(con, start: float, end: float) -> None:
    con.execute("INSERT INTO ais_gaps VALUES (?, ?)", (start, end))
    con.commit()


def last_ais_time(con) -> float | None:
    return con.execute("SELECT MAX(time) FROM ais").fetchone()[0]


def ais_window(con, t0: float, window_s: float, bbox=None) -> pd.DataFrame:
    """AIS within +/- window of t0, optionally only inside bbox = (west, south, east, north)."""
    if bbox is None:
        return pd.read_sql("SELECT * FROM ais WHERE time BETWEEN ? AND ?", con, params=(t0 - window_s, t0 + window_s))
    w, s, e, n = bbox
    return pd.read_sql("SELECT * FROM ais WHERE time BETWEEN ? AND ? AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?",
                       con, params=(t0 - window_s, t0 + window_s, w, e, s, n))


def gap_seconds(con, t0: float, window_s: float) -> float:
    """Total feed outage overlapping [t0-window, t0+window]."""
    lo, hi = t0 - window_s, t0 + window_s
    rows = con.execute("SELECT start, end FROM ais_gaps WHERE end > ? AND start < ?", (lo, hi)).fetchall()
    return float(sum(min(e, hi) - max(s, lo) for s, e in rows))


def save_pass(con, summary: dict, detections: pd.DataFrame) -> None:
    con.execute("DELETE FROM passes WHERE pass_id=?", (summary["pass_id"],))
    con.execute("DELETE FROM detections WHERE pass_id=?", (summary["pass_id"],))
    pd.DataFrame([summary]).to_sql("passes", con, if_exists="append", index=False)
    cols = ["pass_id", "det_id", "lon", "lat", "length_m", "score", "shore_km", "status", "reason", "confidence", "crop"]
    detections[cols].to_sql("detections", con, if_exists="append", index=False)
    con.commit()
