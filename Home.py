import streamlit as st
from utils.db import init_db
from utils.state import has_active_dataset, has_trained_model, get_active_dataset_meta

st.set_page_config(page_title="NoShow AI", page_icon="✅", layout="wide")

# Ensure DB + tables exist on every app start
init_db()

st.title("✅ NoShow AI — Predictive Scheduling")
st.caption("Machine Learning-Based Appointment No-Show Prediction")

st.markdown("""
This tool helps service-based organizations (clinics, salons, repair centers, consulting
firms, etc.) predict which upcoming appointments are likely to be **missed**, so staff can
proactively confirm or follow up.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("How it works")
    st.markdown("""
    1. **Upload** a historical appointments CSV
    2. Explore the data (**Data Analysis**)
    3. Generate model-ready features (**Feature Engineering**)
    4. **Train** a classification model
    5. **Evaluate** its performance
    6. **Predict** no-show risk for new appointments
    7. Review **High-Risk Appointments** and **Business Insights**
    8. **Export** results as CSV
    """)

with col2:
    st.subheader("Current status")
    if has_active_dataset():
        meta = get_active_dataset_meta()
        st.success(f"✅ Dataset loaded: **{meta['filename']}** ({meta['row_count']} rows)")
    else:
        st.warning("⚠️ No dataset uploaded yet.")

    if has_trained_model():
        st.success("✅ A trained model is active.")
    else:
        st.info("ℹ️ No trained model yet.")

    st.divider()
    if not has_active_dataset():
        st.page_link("pages/2_Upload_Dataset.py", label="➡️ Start: Upload Dataset", icon="📤")
    else:
        st.page_link("pages/1_Dashboard.py", label="➡️ Go to Dashboard", icon="📊")
