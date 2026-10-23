"""SQLite persistence for classification history (parameterised queries only)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.utils import load_config, resolve

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    image_filename  TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence      REAL NOT NULL,
    model           TEXT NOT NULL,
    probabilities   TEXT,
    status          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(timestamp);
"""
COLUMNS = ["id", "timestamp", "image_filename", "predicted_class",
           "confidence", "model", "probabilities", "status"]


def default_db_path() -> Path:
    return resolve(load_config()["paths"]["database"])


@contextmanager
def _connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)           # auto-create schema
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    with _connect(db_path):
        pass


def add_prediction(filename: str, predicted_class: str, confidence: float, model: str,
                   probabilities: dict[str, float] | None = None, status: str | None = None,
                   db_path: Path | str | None = None) -> int:
    """Insert one record and return its id."""
    if status is None:
        thr = load_config()["ui"]["low_confidence_threshold"]
        status = "Confident" if confidence >= thr else "Low confidence"
    row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), filename, predicted_class,
           float(confidence), model, json.dumps(probabilities) if probabilities else None, status)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO predictions (timestamp, image_filename, predicted_class, confidence, "
            "model, probabilities, status) VALUES (?, ?, ?, ?, ?, ?, ?)", row)
        return int(cur.lastrowid)


def query_history(category: str | None = None, model: str | None = None,
                  search: str | None = None, newest_first: bool = True,
                  limit: int | None = None, db_path: Path | str | None = None) -> pd.DataFrame:
    sql, params = "SELECT * FROM predictions WHERE 1=1", []
    if category:
        sql += " AND predicted_class = ?"; params.append(category)
    if model:
        sql += " AND model = ?"; params.append(model)
    if search:
        sql += " AND image_filename LIKE ?"; params.append(f"%{search}%")
    sql += " ORDER BY timestamp " + ("DESC" if newest_first else "ASC") + ", id " + \
           ("DESC" if newest_first else "ASC")
    if limit:
        sql += " LIMIT ?"; params.append(int(limit))
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def count_predictions(db_path: Path | str | None = None) -> int:
    with _connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])


def clear_history(db_path: Path | str | None = None) -> int:
    """Delete all records; returns number removed."""
    with _connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.execute("DELETE FROM predictions")
        return int(n)


def export_csv(df: pd.DataFrame) -> bytes:
    out = df.drop(columns=[c for c in ["probabilities"] if c in df.columns]).copy()
    if "timestamp" in out.columns and not out.empty:
        out["timestamp"] = out["timestamp"].astype(str)
    return out.to_csv(index=False).encode("utf-8")
