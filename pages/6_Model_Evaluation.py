import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from utils.state import require_model, load_active_appointments, get_active_model_meta

st.set_page_config(page_title="Model Evaluation", page_icon="📐", layout="wide")
require_model()

meta = get_active_model_meta()
st.title("📐 Model Evaluation")
st.caption(f"Active model: **{meta['model_name']}** · trained {meta['trained_at'][:19]}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{meta['accuracy']:.1%}")
c2.metric("Precision", f"{meta['precision_score']:.1%}")
c3.metric("Recall", f"{meta['recall_score']:.1%}")
c4.metric("F1-score", f"{meta['f1_score']:.1%}")
c5.metric("ROC-AUC", f"{meta['roc_auc']:.3f}")

st.info("💡 Because the goal is catching potential no-shows, **Recall and Precision for the "
        "no-show class** matter more than overall Accuracy alone.")

st.divider()
st.subheader("Confusion Matrix (recomputed on a fresh test split)")

pipe = joblib.load(meta["model_path"])
df = load_active_appointments()

import json
feature_cols = json.loads(meta["feature_columns"])
X = df[feature_cols]
y = df["no_show"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_pred = pipe.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
labels = ["Attended (0)", "No-Show (1)"]

#  reversed y so it reads top-to-bottom like a normal matrix.
z = cm[::-1]
y_labels = labels[::-1]

fig = go.Figure(data=go.Heatmap(
    z=z,
    x=labels,
    y=y_labels,
    colorscale="Blues",
    showscale=True,
    hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
))

annotations = []
for i, row_label in enumerate(y_labels):
    for j, col_label in enumerate(labels):
        value = z[i][j]
        annotations.append(dict(
            x=col_label, y=row_label, text=str(value), showarrow=False,
            font=dict(color="white" if value > z.max() / 2 else "black", size=16),
        ))

fig.update_layout(
    xaxis_title="Predicted", yaxis_title="Actual",
    annotations=annotations, margin=dict(t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)