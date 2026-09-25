import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.state import require_data, load_active_appointments, has_trained_model, get_active_model_meta
from utils.db import get_conn

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# ---- GATE: this is the check you described ----
require_data()
# ------------------------------------------------

df = load_active_appointments()

st.title("📊 Appointment No-Show Prediction")
st.caption("Machine Learning-Based Appointment Risk Assessment")

if has_trained_model():
    meta = get_active_model_meta()
    st.success(f"🟢 Model Active · {meta['model_name']}")
else:
    st.info("No model trained yet — showing raw data stats only. Go to **Train Model** for predictions.")

total = len(df)
no_shows = int(df["no_show"].sum())
attended = total - no_shows

# Pull predictions if a model has been run
with get_conn() as conn:
    pred_row = conn.execute("""
        SELECT COUNT(*) as high_risk, AVG(noshow_probability) as avg_prob
        FROM predictions WHERE risk_level = 'High'
    """).fetchone()
    high_risk = pred_row["high_risk"] or 0
    avg_prob = pred_row["avg_prob"] or 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Appointments", f"{total:,}")
c2.metric("Historical No-Shows", f"{no_shows:,}", f"{(no_shows/total*100):.1f}% of total" if total else None)
c3.metric("High-Risk Appointments", f"{high_risk:,}", "Probability > 60%")
c4.metric("Avg. No-Show Probability", f"{avg_prob*100:.1f}%")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Attended vs No-Show")
    fig = go.Figure(data=[go.Pie(
        labels=["Attended", "No-Show"],
        values=[attended, no_shows],
        hole=0.6,
        marker_colors=["#3B82F6", "#FDE68A"],
    )])
    fig.update_layout(
        annotations=[dict(text=f"{total:,}<br>TOTAL", x=0.5, y=0.5, font_size=18, showarrow=False)],
        showlegend=True, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("No-Show Rate by Lead Time")
    bins = [0, 3, 7, 14, 30, 10000]
    labels = ["0-3d", "4-7d", "8-14d", "15-30d", "30d+"]
    df["lead_bucket"] = pd.cut(df["lead_time_days"], bins=bins, labels=labels, right=True) if "lead_time_days" in df else None
    if df["lead_bucket"].notna().any():
        rate = df.groupby("lead_bucket", observed=True)["no_show"].mean().reset_index()
        rate["no_show"] = rate["no_show"] * 100
        fig2 = px.line(rate, x="lead_bucket", y="no_show", markers=True,
                        labels={"lead_bucket": "Days between booking & appointment", "no_show": "No-show rate (%)"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run Feature Engineering to compute lead time first.")

st.divider()
st.caption("Use the sidebar to Upload new data, run Data Analysis, Train a Model, or Predict individual appointments.")
