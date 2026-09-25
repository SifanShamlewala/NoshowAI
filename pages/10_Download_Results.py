import streamlit as st
import pandas as pd
from utils.state import require_data, load_active_appointments, get_active_batch_id
from utils.db import get_conn

st.set_page_config(page_title="Download Results", page_icon="⬇️", layout="wide")
require_data()

st.title("⬇️ Download Results")

df = load_active_appointments()
batch_id = get_active_batch_id()

with get_conn() as conn:
    preds = pd.read_sql_query("""
        SELECT p.*, a.customer_id, a.service_type, a.appointment_date
        FROM predictions p
        LEFT JOIN appointments a ON p.appointment_id = a.appointment_id
        WHERE a.batch_id = ? OR p.appointment_id IS NULL
    """, conn, params=(batch_id,))

st.subheader("Available exports")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Full appointment dataset**")
    st.caption(f"{len(df)} rows — cleaned + feature-engineered")
    st.download_button("Download appointments.csv", df.to_csv(index=False),
                        file_name="appointments.csv", mime="text/csv", use_container_width=True)

with c2:
    st.markdown("**All predictions**")
    st.caption(f"{len(preds)} rows — probabilities + risk levels")
    st.download_button("Download predictions.csv", preds.to_csv(index=False),
                        file_name="predictions.csv", mime="text/csv", use_container_width=True,
                        disabled=preds.empty)

with c3:
    st.markdown("**Summary report**")
    if not preds.empty:
        summary = pd.DataFrame([{
            "total_appointments": len(df),
            "total_predictions": len(preds),
            "high_risk_count": (preds["risk_level"] == "High").sum(),
            "medium_risk_count": (preds["risk_level"] == "Medium").sum(),
            "low_risk_count": (preds["risk_level"] == "Low").sum(),
            "avg_noshow_probability": preds["noshow_probability"].mean(),
        }])
        st.caption("One-row KPI summary")
        st.download_button("Download summary.csv", summary.to_csv(index=False),
                            file_name="summary.csv", mime="text/csv", use_container_width=True)
    else:
        st.caption("Run predictions first to generate a summary.")
        st.download_button("Download summary.csv", "", disabled=True, use_container_width=True)

st.divider()
st.subheader("Preview: predictions")
st.dataframe(preds.head(50), use_container_width=True)
