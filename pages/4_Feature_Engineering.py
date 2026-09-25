import streamlit as st
from utils.state import require_data, load_active_appointments
from utils.features import FEATURE_COLUMNS

st.set_page_config(page_title="Feature Engineering", page_icon="🧪", layout="wide")
require_data()

st.title("🧪 Feature Engineering")
st.markdown("""
Features below were derived automatically when the dataset was uploaded:

| Feature | Derived from |
|---|---|
| `lead_time_days` | `appointment_date - booking_date` |
| `appointment_hour` | parsed from `appointment_time` |
| `day_of_week`, `is_weekend` | parsed from `appointment_date` |
| `previous_attendance_rate` | `1 - (previous_noshows / previous_appointments)` |
""")

df = load_active_appointments()

st.subheader("Feature preview")
preview_cols = [c for c in FEATURE_COLUMNS if c in df.columns] + ["no_show"]
st.dataframe(df[preview_cols].head(20), use_container_width=True)

st.subheader("Feature summary statistics")
numeric_cols = df[preview_cols].select_dtypes(include="number").columns
st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

st.info(f"These {len(FEATURE_COLUMNS)} features will be used as model inputs in **Train Model**: "
        f"`{', '.join(FEATURE_COLUMNS)}`")
