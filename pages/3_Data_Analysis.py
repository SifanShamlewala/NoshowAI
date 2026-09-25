import streamlit as st
import pandas as pd
import plotly.express as px
from utils.state import require_data, load_active_appointments

st.set_page_config(page_title="Data Analysis", page_icon="📈", layout="wide")
require_data()

st.title("📈 Exploratory Data Analysis")
df = load_active_appointments()
df["no_show_label"] = df["no_show"].map({0: "Attended", 1: "No-Show"})

st.caption(f"Analyzing {len(df):,} appointments from the active dataset.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("No-Show vs Attended")
    fig = px.pie(df, names="no_show_label", hole=0.5)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("No-Show Rate by Service Type")
    rate = df.groupby("service_type")["no_show"].mean().reset_index()
    rate["no_show"] = (rate["no_show"] * 100).round(1)
    fig2 = px.bar(rate.sort_values("no_show", ascending=False), x="service_type", y="no_show",
                  labels={"no_show": "No-show rate (%)", "service_type": "Service Type"})
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("No-Show Rate by Appointment Hour")
    rate_h = df.groupby("appointment_hour")["no_show"].mean().reset_index()
    rate_h["no_show"] = (rate_h["no_show"] * 100).round(1)
    fig3 = px.line(rate_h, x="appointment_hour", y="no_show", markers=True,
                   labels={"no_show": "No-show rate (%)", "appointment_hour": "Hour of day"})
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("No-Show Rate by Day of Week")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rate_d = df.groupby("day_of_week")["no_show"].mean().reindex(order).reset_index()
    rate_d["no_show"] = (rate_d["no_show"] * 100).round(1)
    fig4 = px.bar(rate_d, x="day_of_week", y="no_show",
                  labels={"no_show": "No-show rate (%)", "day_of_week": "Day"})
    st.plotly_chart(fig4, use_container_width=True)

col5, col6 = st.columns(2)

with col5:
    st.subheader("Lead Time vs No-Show Rate")
    bins = [0, 3, 7, 14, 30, 10000]
    labels = ["0-3d", "4-7d", "8-14d", "15-30d", "30d+"]
    df["lead_bucket"] = pd.cut(df["lead_time_days"], bins=bins, labels=labels)
    rate_l = df.groupby("lead_bucket", observed=True)["no_show"].mean().reset_index()
    rate_l["no_show"] = (rate_l["no_show"] * 100).round(1)
    fig5 = px.line(rate_l, x="lead_bucket", y="no_show", markers=True,
                   labels={"no_show": "No-show rate (%)", "lead_bucket": "Lead time"})
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("Reminder Sent vs Attendance")
    rate_r = df.groupby("reminder_sent")["no_show"].mean().reset_index()
    rate_r["reminder_sent"] = rate_r["reminder_sent"].map({0: "No Reminder", 1: "Reminder Sent"})
    rate_r["no_show"] = (rate_r["no_show"] * 100).round(1)
    fig6 = px.bar(rate_r, x="reminder_sent", y="no_show",
                  labels={"no_show": "No-show rate (%)", "reminder_sent": ""})
    st.plotly_chart(fig6, use_container_width=True)

st.divider()
st.subheader("Previous Attendance Behavior")
fig7 = px.scatter(df, x="previous_attendance_rate", y="lead_time_days", color="no_show_label",
                   labels={"previous_attendance_rate": "Previous attendance rate", "lead_time_days": "Lead time (days)"},
                   opacity=0.6)
st.plotly_chart(fig7, use_container_width=True)
