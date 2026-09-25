import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
from utils.db import get_conn
from utils.features import validate_columns, clean_dataframe, engineer_features, REQUIRED_COLUMNS

st.set_page_config(page_title="Upload Dataset", page_icon="📤", layout="wide")

st.title("📤 Upload Dataset")
st.markdown("Upload a historical appointments CSV. This becomes the active dataset used by every "
            "other page (Data Analysis, Feature Engineering, Train Model, etc.).")

with st.expander("📋 Required columns", expanded=False):
    st.code(", ".join(REQUIRED_COLUMNS))
    st.markdown("""
    - `booking_date`, `appointment_date`: any standard date format (e.g. `2026-09-01`)
    - `appointment_time`: e.g. `14:30` or `2:30 PM`
    - `reminder_sent`, `no_show`: `0`/`1`, `Yes`/`No`, or `True`/`False`
    - `previous_appointments`, `previous_noshows`: integers
    """)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read this file as CSV: {e}")
        st.stop()

    st.subheader("Preview (raw upload)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    is_valid, missing = validate_columns(raw_df)

    if not is_valid:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        st.stop()

    st.success("✅ All required columns present.")

    # Clean + engineer
    cleaned = clean_dataframe(raw_df)
    rows_dropped = len(raw_df) - len(cleaned)
    featured = engineer_features(cleaned)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows uploaded", len(raw_df))
    c2.metric("Rows dropped (dupes/bad dates)", rows_dropped)
    c3.metric("Rows ready to import", len(featured))

    st.subheader("Preview (cleaned + feature-engineered)")
    st.dataframe(featured.head(10), use_container_width=True)

    dataset_name = st.text_input("Name this dataset batch (optional)", value=uploaded_file.name)

    if st.button("✅ Confirm & Import to Database", type="primary"):
        batch_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        with get_conn() as conn:
            c = conn.cursor()

            # Deactivate any previous dataset (v1 = single active dataset at a time)
            c.execute("UPDATE dataset_metadata SET is_active = 0")

            c.execute("""
                INSERT INTO dataset_metadata (batch_id, filename, row_count, uploaded_at, is_active)
                VALUES (?, ?, ?, ?, 1)
            """, (batch_id, dataset_name, len(featured), now))

            insert_cols = [
                "customer_id", "service_type", "booking_date", "appointment_date",
                "appointment_time", "lead_time_days", "appointment_hour", "day_of_week",
                "is_weekend", "previous_appointments", "previous_noshows",
                "previous_attendance_rate", "reminder_sent", "no_show",
            ]

            records = []
            for _, row in featured.iterrows():
                records.append((
                    batch_id,
                    str(row.get("customer_id", "")),
                    str(row.get("service_type", "")),
                    str(row.get("booking_date")),
                    str(row.get("appointment_date")),
                    str(row.get("appointment_time", "")),
                    int(row.get("lead_time_days", 0)),
                    int(row.get("appointment_hour", 12)),
                    str(row.get("day_of_week", "")),
                    int(row.get("is_weekend", 0)),
                    int(row.get("previous_appointments", 0)),
                    int(row.get("previous_noshows", 0)),
                    float(row.get("previous_attendance_rate", 1.0)),
                    int(row.get("reminder_sent", 0)),
                    int(row.get("no_show", 0)),
                ))

            placeholders = ", ".join(["?"] * (len(insert_cols) + 1))  # +1 for batch_id
            c.executemany(
                f"""INSERT INTO appointments
                    (batch_id, customer_id, service_type, booking_date, appointment_date,
                     appointment_time, lead_time_days, appointment_hour, day_of_week,
                     is_weekend, previous_appointments, previous_noshows,
                     previous_attendance_rate, reminder_sent, no_show)
                    VALUES ({placeholders})""",
                records
            )

            conn.commit()

        st.success(f"🎉 Dataset imported and set as active (batch `{batch_id}`, {len(featured)} rows).")
        st.balloons()
        st.page_link("pages/1_Dashboard.py", label="➡️ Go to Dashboard", icon="📊")
else:
    st.info("No file selected yet.")
