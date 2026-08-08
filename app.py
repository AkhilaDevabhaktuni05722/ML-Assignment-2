
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="🤖",
    layout="wide"
)

MODEL_DIR = Path(__file__).parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl"
}

st.title("🤖 Machine Learning Classification Model Comparison")
st.write(
    "Interactive Streamlit application for evaluating classification models "
    "using the Breast Cancer Wisconsin Diagnostic dataset."
)

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Select a classification model:",
    list(MODEL_FILES.keys())
)

st.sidebar.header("Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload test data CSV",
    type=["csv"]
)

@st.cache_resource
def load_model(model_name):
    with open(MODEL_DIR / MODEL_FILES[model_name], "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_default_test_data():
    return pd.read_csv(Path(__file__).parent / "test_data.csv")

# Load uploaded test data or default test data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded test data loaded successfully.")
else:
    df = load_default_test_data()
    st.info("No file uploaded. The included test_data.csv is being used.")

TARGET = "diagnosis"

if TARGET not in df.columns:
    st.error(
        "The uploaded CSV must contain a 'diagnosis' column as the target."
    )
    st.stop()

X = df.drop(columns=[TARGET])
y = df[TARGET]

model = load_model(selected_model)

# Make sure feature columns match training columns.
# Saved models expect the 30 WDBC features.
try:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(
        "The uploaded CSV does not contain the expected feature columns "
        "or their structure does not match the training data."
    )
    st.exception(e)
    st.stop()

# Metrics
accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
mcc = matthews_corrcoef(y, y_pred)

st.subheader(f"Performance: {selected_model}")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC", f"{auc:.4f}")
c3.metric("Precision", f"{precision:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)
st.dataframe(cm_df, use_container_width=True)

st.subheader("Classification Report")
report = classification_report(
    y, y_pred, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(4), use_container_width=True)

st.subheader("Uploaded/Test Data Preview")
st.dataframe(df.head(20), use_container_width=True)

st.caption(
    "Dataset: Breast Cancer Wisconsin Diagnostic (WDBC-compatible). "
    "Target: diagnosis (0 = malignant, 1 = benign)."
)
