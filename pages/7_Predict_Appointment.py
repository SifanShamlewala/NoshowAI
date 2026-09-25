import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime, date, time as dtime

from utils.state import require_model, load_active_appointments, get_active_model_meta
from utils.features import risk_level
from utils.db import get_conn

st.set_page_config(page_title="Predict Appointment", page_icon="🔮", layout="wide")
require_model()

meta = get_active_model_meta()
pipe = joblib.load(meta["model_path"])
df = load_active_appointments()
service_types = sorted(df["service_type"].dropna().unique().tolist()) or ["General"]

st.title("🔮 Predict Appointment Risk")

col_form, col_result = st.columns([1, 1])

with col_form:
    st.subheader("Predict Appointment Risk")
    st.caption("Single appointment")

    c1, c2 = st.columns(2)
    with c1:
        service_type = st.selectbox("Service Type", service_types)
    with c2:
        lead_time = st.number_input("Lead Time (days)", min_value=0, value=21)

    c3, c4 = st.columns(2)
    with c3:
        appt_date = st.date_input("Appointment Date", value=date.today())
    with c4:
        appt_time = st.time_input("Appointment Time", value=dtime(14, 30))

    c5, c6 = st.columns(2)
    with c5:
        prev_appts = st.number_input("Previous Appointments", min_value=0, value=6)
    with c6:
        prev_noshows = st.number_input("Previous No-Shows", min_value=0, value=2)

    reminder_sent = st.toggle("SMS reminder scheduled 24h prior", value=True)

    predict_clicked = st.button("Predict No-Show Risk", type="primary", use_container_width=True)

with col_result:
    st.subheader("Prediction Result")

    if predict_clicked:
        is_weekend = int(appt_date.weekday() >= 5)
        day_of_week = appt_date.strftime("%A")
        attendance_rate = 1 - (prev_noshows / prev_appts) if prev_appts > 0 else 1.0

        row = pd.DataFrame([{
            "lead_time_days": lead_time,
            "appointment_hour": appt_time.hour,
            "is_weekend": is_weekend,
            "previous_appointments": prev_appts,
            "previous_noshows": prev_noshows,
            "previous_attendance_rate": attendance_rate,
            "reminder_sent": int(reminder_sent),
            "service_type": service_type,
            "day_of_week": day_of_week,
        }])

        feature_cols = json.loads(meta["feature_columns"])
        prob = float(pipe.predict_proba(row[feature_cols])[0, 1])
        risk = risk_level(prob)

        color = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[risk]

        st.metric("NO-SHOW PROB.", f"{prob*100:.1f}%")
        st.markdown(f"### {color} **{risk} Risk**")

        if risk == "High":
            action = "Confirm appointment with customer. Consider a personal call in addition to the automated reminder."
        elif risk == "Medium":
            action = "Send an additional reminder closer to the appointment date."
        else:
            action = "No special action needed — standard reminder is sufficient."

        st.info(f"**SUGGESTED ACTION**\n\n{action}")

        st.divider()
        d1, d2, d3 = st.columns(3)
        d1.metric("Lead time", f"{lead_time} days")
        d2.metric("Prior no-shows", f"{prev_noshows} of {prev_appts}")
        d3.metric("Reminder sent", "Yes" if reminder_sent else "No")

        # Log this ad-hoc prediction
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO predictions (appointment_id, model_id, noshow_probability, risk_level, predicted_at, source)
                VALUES (NULL, ?, ?, ?, ?, 'manual')
            """, (meta["model_id"], prob, risk, datetime.now().isoformat()))
            conn.commit()
    else:
        st.caption("Fill in the appointment details and click **Predict No-Show Risk**.")
