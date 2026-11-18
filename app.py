import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="wide")

st.title("🕵️ Fake Job Posting Detection")
st.caption("Binary classification using Logistic Regression")
st.write(
    "This app follows the uploaded notebook: it trains a Logistic Regression "
    "pipeline using **telecommuting**, **has_company_logo**, and **has_questions**."
)

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, on_bad_lines="skip")
    return pd.read_csv("fake_job_postings.csv", on_bad_lines="skip")

def prepare_data(raw):
    required = ["telecommuting", "has_company_logo", "has_questions", "fraudulent"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(missing))

    data = raw.dropna(subset=["fraudulent"]).copy()
    feature_cols = required[:3]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)
    data["fraudulent"] = pd.to_numeric(data["fraudulent"], errors="coerce")
    data = data.dropna(subset=["fraudulent"])
    data["fraudulent"] = data["fraudulent"].astype(int)
    data = data[data["fraudulent"].isin([0, 1])]
    if data.empty or data["fraudulent"].nunique() < 2:
        raise ValueError("The dataset must contain both target classes: 0 (real) and 1 (fraudulent).")
    return data, feature_cols

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload fake_job_postings.csv", type=["csv"])
    st.info("If no file is uploaded, the app looks for fake_job_postings.csv in this project folder.")
    st.divider()
    st.write("Model: Logistic Regression")
    st.write("Test size: 20%")
    st.write("Random state: 42")

try:
    raw_df = load_data(uploaded)
    df, features = prepare_data(raw_df)
except FileNotFoundError:
    st.error("Dataset not found. Upload fake_job_postings.csv using the sidebar, or place it beside app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load/prepare dataset: {exc}")
    st.stop()

# Same split and pipeline approach as the notebook
X = df[features]
y = df["fraudulent"]
class_counts = y.value_counts()
if class_counts.min() < 2:
    st.error("Each class needs at least 2 rows to create a stratified train/test split.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

# Overview
st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total rows used", f"{len(df):,}")
c2.metric("Features", len(features))
c3.metric("Training rows", f"{len(X_train):,}")
c4.metric("Testing rows", f"{len(X_test):,}")

left, right = st.columns(2)
with left:
    st.markdown("**Class distribution**")
    dist = y.value_counts().reindex([0, 1], fill_value=0)
    chart_df = pd.DataFrame({"Job type": ["Real (0)", "Fraudulent (1)"], "Count": dist.values})
    st.bar_chart(chart_df.set_index("Job type"))
with right:
    st.markdown("**Data quality**")
    st.write(f"Missing values in original dataset: **{int(raw_df.isna().sum().sum()):,}**")
    st.write(f"Duplicate rows in original dataset: **{int(raw_df.duplicated().sum()):,}**")
    st.write("Features used:", ", ".join(features))

st.divider()
st.subheader("Model evaluation")
metrics = [
    ("Accuracy", accuracy_score(y_test, predictions)),
    ("Precision", precision_score(y_test, predictions, zero_division=0)),
    ("Recall", recall_score(y_test, predictions, zero_division=0)),
    ("F1-score", f1_score(y_test, predictions, zero_division=0)),
]
metric_cols = st.columns(4)
for col, (label, value) in zip(metric_cols, metrics):
    col.metric(label, f"{value:.3f}")

cm = confusion_matrix(y_test, predictions, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 3.5))
im = ax.imshow(cm)
ax.set_xticks([0, 1], labels=["Predicted real", "Predicted fraud"])
ax.set_yticks([0, 1], labels=["Actual real", "Actual fraud"])
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix")
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, str(value), ha="center", va="center")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
st.pyplot(fig)
plt.close(fig)

with st.expander("Classification report"):
    report = classification_report(y_test, predictions, target_names=["Real (0)", "Fraudulent (1)"], zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

with st.expander("Preview cleaned dataset"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()
st.subheader("🔎 Check a job posting")
st.write("Enter the three values used by this notebook. The prediction is based only on these fields, not the full job description.")
with st.form("prediction_form"):
    p1, p2, p3 = st.columns(3)
    telecommuting = p1.selectbox("Telecommuting?", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")
    company_logo = p2.selectbox("Company logo present?", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")
    questions = p3.selectbox("Screening questions present?", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")
    submitted = st.form_submit_button("Predict job type", type="primary")

if submitted:
    sample = pd.DataFrame([{
        "telecommuting": telecommuting,
        "has_company_logo": company_logo,
        "has_questions": questions
    }], columns=features)
    result = int(model.predict(sample)[0])
    probability = model.predict_proba(sample)[0][list(model.named_steps["classifier"].classes_).index(1)]
    if result == 1:
        st.error(f"Model prediction: LIKELY FRAUDULENT (1) — fraud probability {probability:.1%}")
    else:
        st.success(f"Model prediction: LIKELY REAL (0) — fraud probability {probability:.1%}")
    st.warning("Educational prototype only. These three metadata features are not enough to reliably determine whether a job is genuine.")

st.caption("Target labels: 0 = real job posting, 1 = fraudulent job posting.")
