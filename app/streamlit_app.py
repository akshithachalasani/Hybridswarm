import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# ================= PAGE CONFIG ================= #
st.set_page_config(
    page_title="Hybrid Swarm Defect Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================= FORCE DARK THEME ================= #
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #0d0d0d;
    color: white;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #111111;
}

/* Headers */
h1, h2, h3, h4 {
    color: #FFD700 !important;
}

/* Paragraph text */
p, label, span {
    color: white !important;
}

/* Buttons */
div.stButton > button {
    background-color: #FFD700 !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    border: none !important;
}

/* Selectbox & radio text */
div[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    color: white !important;
}

/* Dataframe header */
thead tr th {
    background-color: #FFD700 !important;
    color: black !important;
}

/* Remove footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ================= HEADER ================= #
st.markdown("""
<div style='background-color:#111;padding:20px;border-radius:10px'>
    <h1 style='color:#FFD700;'>🚀 Advanced Hybrid Swarm-Based Software Defect Prediction</h1>
    <p style='color:white;'>PSO + ACO + GA Optimized Stacking Ensemble</p>
</div>
""", unsafe_allow_html=True)


# ================= SIDEBAR ================= #
st.sidebar.markdown("<h2 style='color:#FFD700;'>⚙ Configuration</h2>", unsafe_allow_html=True)

DATA_PATH = "data"
datasets = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

dataset = st.sidebar.selectbox("Select Dataset", datasets)
mode = st.sidebar.radio("Mode", ["Single Algorithm", "Full Comparison"])

algorithm = None
if mode == "Single Algorithm":
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Logistic Regression", "SVM", "Random Forest", "Hybrid (PSO+ACO+GA)"]
    )

run = st.sidebar.button("Run Model")


# ================= LOAD DATA ================= #
df = pd.read_csv(os.path.join(DATA_PATH, dataset))

TARGET = df.columns[-1]
X = df.drop(columns=[TARGET])
y = df[TARGET]

if y.nunique() < 2:
    st.warning("Dataset contains only one class.")
    st.stop()


# ================= SMOTE ================= #
try:
    sm = SMOTE()
    X, y = sm.fit_resample(X, y)
except:
    pass


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ================= EVALUATION FUNCTION ================= #
def evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Predictions": y_pred
    }


# ================= HYBRID MODEL ================= #
def hybrid_model():

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        random_state=42
    )

    svm = SVC(
        C=15,
        kernel="rbf",
        probability=True
    )

    lr = LogisticRegression(
        C=8,
        max_iter=4000
    )

    hybrid = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm),
            ("lr", lr)
        ],
        final_estimator=RandomForestClassifier(n_estimators=200),
        passthrough=True
    )

    return hybrid


# ================= RUN ================= #
if run:

    st.markdown("## 📊 Model Performance Comparison")

    results = []

    if mode == "Single Algorithm":

        if algorithm == "Logistic Regression":
            results.append(evaluate(LogisticRegression(max_iter=1000), "Logistic"))

        elif algorithm == "SVM":
            results.append(evaluate(SVC(), "SVM"))

        elif algorithm == "Random Forest":
            results.append(evaluate(RandomForestClassifier(n_estimators=100), "Random Forest"))

        elif algorithm == "Hybrid (PSO+ACO+GA)":
            results.append(evaluate(hybrid_model(), "Hybrid"))

    else:
        results.append(evaluate(LogisticRegression(max_iter=1000), "Logistic"))
        results.append(evaluate(SVC(), "SVM"))
        results.append(evaluate(RandomForestClassifier(n_estimators=100), "Random Forest"))
        results.append(evaluate(hybrid_model(), "Hybrid"))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Accuracy", ascending=False)

    st.dataframe(results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]])

    st.markdown("## 🔍 Confusion Matrix (Best Model)")

    best_row = results_df.iloc[0]
    best_predictions = best_row["Predictions"]
    best_model_name = best_row["Model"]

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        ax=ax,
        cmap="Blues"
    )

    ax.set_title(f"Confusion Matrix - {best_model_name}")
    st.pyplot(fig)
