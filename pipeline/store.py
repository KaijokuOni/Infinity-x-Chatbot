"""SQLite persistence for patient cases shown on the doctor dashboard.

One row per browser session: the structured case (chief complaint, symptoms,
possible conditions, …) plus the full Bangla transcript. Updated after every
turn so the dashboard always reflects the latest state.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "cases.db"

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def init_db() -> None:
    with _lock:
        c = _connect()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                session_id          TEXT PRIMARY KEY,
                created_at          REAL,
                updated_at          REAL,
                patient_name        TEXT,
                patient_age         TEXT,
                chief_complaint     TEXT,
                symptoms            TEXT,
                duration            TEXT,
                severity            TEXT,
                tests_done          TEXT,
                possible_conditions TEXT,
                summary             TEXT,
                transcript          TEXT,
                turns               INTEGER
            )
            """
        )
        # Migrate older DBs that predate the patient name/age columns.
        cols = {r["name"] for r in c.execute("PRAGMA table_info(cases)")}
        for col in ("patient_name", "patient_age"):
            if col not in cols:
                c.execute(f"ALTER TABLE cases ADD COLUMN {col} TEXT")
        c.commit()


def save_case(
    session_id: str,
    structured: dict,
    transcript: list[dict],
    name: str = "",
    age: str = "",
) -> None:
    now = time.time()
    with _lock:
        c = _connect()
        row = c.execute(
            "SELECT created_at FROM cases WHERE session_id=?", (session_id,)
        ).fetchone()
        created = row["created_at"] if row else now
        c.execute(
            """
            INSERT INTO cases (
                session_id, created_at, updated_at, patient_name, patient_age,
                chief_complaint, symptoms, duration, severity, tests_done,
                possible_conditions, summary, transcript, turns
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at          = excluded.updated_at,
                patient_name        = excluded.patient_name,
                patient_age         = excluded.patient_age,
                chief_complaint     = excluded.chief_complaint,
                symptoms            = excluded.symptoms,
                duration            = excluded.duration,
                severity            = excluded.severity,
                tests_done          = excluded.tests_done,
                possible_conditions = excluded.possible_conditions,
                summary             = excluded.summary,
                transcript          = excluded.transcript,
                turns               = excluded.turns
            """,
            (
                session_id,
                created,
                now,
                name,
                age,
                structured.get("chief_complaint", ""),
                json.dumps(structured.get("symptoms", []), ensure_ascii=False),
                structured.get("duration", ""),
                structured.get("severity", ""),
                json.dumps(structured.get("tests_done", []), ensure_ascii=False),
                json.dumps(structured.get("possible_conditions", []), ensure_ascii=False),
                structured.get("summary", ""),
                json.dumps(transcript, ensure_ascii=False),
                len(transcript),
            ),
        )
        c.commit()


def _row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "session_id": r["session_id"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "patient_name": r["patient_name"] or "",
        "patient_age": r["patient_age"] or "",
        "chief_complaint": r["chief_complaint"] or "",
        "symptoms": json.loads(r["symptoms"] or "[]"),
        "duration": r["duration"] or "",
        "severity": r["severity"] or "",
        "tests_done": json.loads(r["tests_done"] or "[]"),
        "possible_conditions": json.loads(r["possible_conditions"] or "[]"),
        "summary": r["summary"] or "",
        "transcript": json.loads(r["transcript"] or "[]"),
        "turns": r["turns"] or 0,
    }


def list_cases() -> list[dict]:
    with _lock:
        c = _connect()
        rows = c.execute(
            "SELECT * FROM cases ORDER BY updated_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]
