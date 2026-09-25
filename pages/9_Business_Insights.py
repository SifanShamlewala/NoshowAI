import streamlit as st
import pandas as pd
import plotly.express as px
from utils.state import require_data, load_active_appointments

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide")
require_data()

st.title("💡 Business Insights")
df = load_active_appointments()

st.subheader("1. Which services have higher predicted no-show rates?")
svc = df.groupby("service_type")["no_show"].agg(["mean", "count"]).reset_index()
svc.columns = ["Service Type", "No-Show Rate", "Appointments"]
svc["No-Show Rate"] = (svc["No-Show Rate"] * 100).round(1)
st.dataframe(svc.sort_values("No-Show Rate", ascending=False), use_container_width=True)

st.subheader("2. Which appointment periods have greater risk?")
hr = df.groupby("appointment_hour")["no_show"].mean().reset_index()
hr["no_show"] = (hr["no_show"] * 100).round(1)
riskiest_hours = hr.sort_values("no_show", ascending=False).head(3)
st.markdown(f"Riskiest hours: **{', '.join(str(int(h)) + ':00' for h in riskiest_hours['appointment_hour'])}**")
fig = px.bar(hr, x="appointment_hour", y="no_show", labels={"no_show": "No-show rate (%)"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Which customer behavior patterns correlate with no-shows?")
corr_cols = ["previous_attendance_rate", "previous_noshows", "reminder_sent", "lead_time_days", "no_show"]
corr = df[corr_cols].corr()["no_show"].drop("no_show").sort_values(key=abs, ascending=False)
st.dataframe(corr.rename("Correlation with No-Show").to_frame(), use_container_width=True)
st.caption("Negative = higher value associated with LOWER no-show rate. Positive = higher value associated with MORE no-shows.")

st.subheader("4. How does appointment lead time affect attendance?")
bins = [0, 3, 7, 14, 30, 10000]
labels = ["0-3d", "4-7d", "8-14d", "15-30d", "30d+"]
df["lead_bucket"] = pd.cut(df["lead_time_days"], bins=bins, labels=labels)
lead = df.groupby("lead_bucket", observed=True)["no_show"].mean().reset_index()
lead["no_show"] = (lead["no_show"] * 100).round(1)
fig2 = px.line(lead, x="lead_bucket", y="no_show", markers=True, labels={"no_show": "No-show rate (%)"})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("5. Which appointments require additional confirmation?")
candidates = df[(df["previous_noshows"] >= 2) | (df["reminder_sent"] == 0)]
st.markdown(f"**{len(candidates)}** appointments have ≥2 prior no-shows OR no reminder scheduled — "
            "good candidates for a manual confirmation call.")
show_cols = [c for c in ["appointment_id", "customer_id", "service_type", "appointment_date",
                          "previous_noshows", "reminder_sent"] if c in candidates.columns]
st.dataframe(candidates[show_cols].head(50), use_container_width=True)
