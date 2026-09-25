"""
Shared state / gating helpers.
Every page that needs data or a trained model calls require_data() / require_model()
at the very top, BEFORE doing any work. This is what implements:

    "check if data exists -> if not, tell user to upload first"

for every page in the app, in one consistent place.
"""
import streamlit as st
import pandas as pd
from utils.db import get_conn


def has_active_dataset() -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM dataset_metadata WHERE is_active = 1"
        ).fetchone()
        return row["cnt"] > 0


def get_active_batch_id():
    with get_conn() as conn:
        row = conn.execute(
            "SELECT batch_id FROM dataset_metadata WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        return row["batch_id"] if row else None


def get_active_dataset_meta():
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM dataset_metadata WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def load_active_appointments() -> pd.DataFrame:
    batch_id = get_active_batch_id()
    if not batch_id:
        return pd.DataFrame()
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM appointments WHERE batch_id = ?", conn, params=(batch_id,)
        )
    return df


def has_trained_model() -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM model_registry WHERE is_active = 1"
        ).fetchone()
        return row["cnt"] > 0


def get_active_model_meta():
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM model_registry WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def require_data():
    """Call at the top of any page that needs an uploaded dataset."""
    if not has_active_dataset():
        st.warning("⚠️ No dataset found yet.")
        st.info("Head to **Upload Dataset** and upload a historical appointments CSV to get started.")
        try:
            st.page_link("pages/2_Upload_Dataset.py", label="➡️ Go to Upload Dataset", icon="📤")
        except Exception:
            pass
        st.stop()


def require_model():
    """Call at the top of any page that needs a trained model (Evaluate/Predict/High-Risk)."""
    require_data()
    if not has_trained_model():
        st.warning("⚠️ No trained model yet.")
        st.info("Head to **Train Model** to train a model on the uploaded dataset first.")
        try:
            st.page_link("pages/5_Train_Model.py", label="➡️ Go to Train Model", icon="🤖")
        except Exception:
            pass
        st.stop()
