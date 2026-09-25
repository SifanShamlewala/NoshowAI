import streamlit as st
import joblib
import json
from datetime import datetime

from utils.state import require_model, load_active_appointments, get_active_model_meta
from utils.features import risk_level
from utils.db import get_conn

st.set_page_config(page_title="High-Risk Appointments", page_icon="🚨", layout="wide")
require_model()

meta = get_active_model_meta()
pipe = joblib.load(meta["model_path"])
df = load_active_appointments()
feature_cols = json.loads(meta["feature_columns"])

st.title("🚨 High-Risk Appointments")

if st.button("🔄 Score all appointments in active dataset", type="primary"):
    probs = pipe.predict_proba(df[feature_cols])[:, 1]
    df["noshow_probability"] = probs
    df["risk_level"] = df["noshow_probability"].apply(risk_level)

    with get_conn() as conn:
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.executemany(
            """INSERT INTO predictions (appointment_id, model_id, noshow_probability, risk_level, predicted_at, source)
               VALUES (?, ?, ?, ?, ?, 'batch')""",
            [(int(r.appointment_id), meta["model_id"], float(r.noshow_probability), r.risk_level, now)
             for r in df.itertuples()]
        )
        conn.commit()

    st.session_state["scored_df"] = df
    st.success(f"Scored {len(df)} appointments.")

df_scored = st.session_state.get("scored_df")

if df_scored is not None:
    threshold = st.slider("Risk threshold (%) to flag as High", 0, 100, 60) / 100
    flagged = df_scored[df_scored["noshow_probability"] >= threshold].sort_values(
        "noshow_probability", ascending=False
    )

    st.metric("Appointments above threshold", len(flagged))

    show_cols = ["appointment_id", "customer_id", "service_type", "appointment_date",
                 "appointment_time", "lead_time_days", "previous_noshows",
                 "reminder_sent", "noshow_probability", "risk_level"]
    show_cols = [c for c in show_cols if c in flagged.columns]

    st.dataframe(
        flagged[show_cols].style.format({"noshow_probability": "{:.1%}"}),
        use_container_width=True, height=500
    )
else:
    st.info("Click **Score all appointments** to generate risk levels for the active dataset.")
