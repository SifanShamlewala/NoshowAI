"""
Database layer for the No-Show Prediction app.
Single SQLite file, WAL mode for concurrent Streamlit reruns.
"""
import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "noshow.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist. Safe to call on every app start."""
    with get_conn() as conn:
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            batch_id TEXT PRIMARY KEY,
            filename TEXT,
            row_count INTEGER,
            uploaded_at TEXT,
            is_active INTEGER DEFAULT 0
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT,
            customer_id TEXT,
            service_type TEXT,
            booking_date TEXT,
            appointment_date TEXT,
            appointment_time TEXT,
            lead_time_days INTEGER,
            appointment_hour INTEGER,
            day_of_week TEXT,
            is_weekend INTEGER,
            previous_appointments INTEGER,
            previous_noshows INTEGER,
            previous_attendance_rate REAL,
            reminder_sent INTEGER,
            no_show INTEGER,
            FOREIGN KEY (batch_id) REFERENCES dataset_metadata(batch_id)
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            trained_on_batch_id TEXT,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            roc_auc REAL,
            model_path TEXT,
            feature_columns TEXT,
            is_active INTEGER DEFAULT 0,
            trained_at TEXT
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            appointment_id INTEGER,
            model_id INTEGER,
            noshow_probability REAL,
            risk_level TEXT,
            predicted_at TEXT,
            source TEXT DEFAULT 'batch'
        )
        """)

        conn.commit()


def reset_db():
    """Danger: wipes all tables. Used only from an explicit admin action."""
    with get_conn() as conn:
        c = conn.cursor()
        for t in ["predictions", "model_registry", "appointments", "dataset_metadata"]:
            c.execute(f"DROP TABLE IF EXISTS {t}")
        conn.commit()
    init_db()
