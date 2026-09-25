import streamlit as st
import pandas as pd
import json
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils.state import require_data, load_active_appointments, get_active_batch_id
from utils.features import FEATURE_COLUMNS
from utils.db import get_conn

st.set_page_config(page_title="Train Model", page_icon="🤖", layout="wide")
require_data()

st.title("🤖 Train Model")

df = load_active_appointments()

if df["no_show"].nunique() < 2:
    st.error("The active dataset needs both attended AND no-show examples (`no_show` = 0 and 1) to train a classifier.")
    st.stop()

NUMERIC_FEATURES = ["lead_time_days", "appointment_hour", "is_weekend",
                     "previous_appointments", "previous_noshows",
                     "previous_attendance_rate", "reminder_sent"]
CATEGORICAL_FEATURES = ["service_type", "day_of_week"]

ALGORITHMS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

st.markdown("Select one or more algorithms to train and compare. The best one (by F1-score on the "
            "no-show class) is set as the **active model** used for predictions.")

selected = st.multiselect("Algorithms", list(ALGORITHMS.keys()),
                           default=["Logistic Regression", "Random Forest"])
test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

if st.button("🚀 Train Selected Models", type="primary", disabled=(len(selected) == 0)):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["no_show"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    results = []
    trained_pipelines = {}

    progress = st.progress(0, text="Training...")
    for i, name in enumerate(selected):
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", ALGORITHMS[name]),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        results.append(metrics)
        trained_pipelines[name] = pipe
        progress.progress((i + 1) / len(selected), text=f"Trained {name}")

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    st.subheader("Comparison")
    st.dataframe(results_df.style.format({c: "{:.3f}" for c in results_df.columns if c != "Model"}),
                 use_container_width=True)

    best_name = results_df.iloc[0]["Model"]
    best_pipe = trained_pipelines[best_name]
    best_metrics = results_df.iloc[0]

    st.success(f"🏆 Best model: **{best_name}** (F1 = {best_metrics['F1']:.3f}) — set as active model.")

    # Persist model to disk
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f"model_{best_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(best_pipe, model_path)

    batch_id = get_active_batch_id()
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE model_registry SET is_active = 0")
        c.execute("""
            INSERT INTO model_registry
                (model_name, trained_on_batch_id, accuracy, precision_score, recall_score,
                 f1_score, roc_auc, model_path, feature_columns, is_active, trained_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """, (
            best_name, batch_id,
            float(best_metrics["Accuracy"]), float(best_metrics["Precision"]),
            float(best_metrics["Recall"]), float(best_metrics["F1"]), float(best_metrics["ROC-AUC"]),
            model_path, json.dumps(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
            datetime.now().isoformat(),
        ))
        conn.commit()

    st.session_state["last_train_results"] = results_df.to_dict("records")
    st.page_link("pages/6_Model_Evaluation.py", label="➡️ Go to Model Evaluation", icon="📐")
