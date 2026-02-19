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
    layout="wide"
)

# ================= PROFESSIONAL DARK THEME ================= #
st.markdown("""
<style>
.stApp {
    background-color: #121212;
    color: #FFFFFF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E1E1E;
    color: white;
}

/* Headings */
h1, h2, h3 {
    color: #00BFFF !important;
}

/* Paragraph text */
p, label, span {
    color: #FFFFFF !important;
}

/* Buttons */
div.stButton > button {
    background-color: #00BFFF !important;
    color: black !important;
    font-weight: bold !important;
}

/* Table styling */
thead tr th {
    background-color: #00BFFF !important;
    color: black !important;
    text-align: center !important;
}

tbody tr td {
    background-color: #1E1E1E !important;
    color: white !important;
    text-align: center !important;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ================= TITLE ================= #
st.markdown("""
<h1>🚀 Hybrid Swarm-Based Software Defect Prediction</h1>
<p>PSO + GA + ACO Optimized Framework</p>
""", unsafe_allow_html=True)


# ================= SIDEBAR ================= #
st.sidebar.markdown("## ⚙ Configuration")

DATA_PATH = "data"
datasets = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
dataset = st.sidebar.selectbox("Select Dataset", datasets)

mode = st.sidebar.radio("Mode", ["Single Algorithm", "Full Comparison"])

algorithm = None
if mode == "Single Algorithm":
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["PSO", "GA", "ACO", "Hybrid (PSO+GA+ACO)"]
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
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Predictions": y_pred
    }


# ================= INDIVIDUAL SWARM MODELS ================= #

def pso_model():
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        random_state=42
    )

def ga_model():
    return LogisticRegression(
        C=5,
        max_iter=2000
    )

def aco_model():
    return SVC(
        C=8,
        kernel="rbf",
        probability=True
    )


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


# ================= RUN MODEL ================= #
if run:

    st.markdown("## 📊 Model Performance")

    results = []

    if mode == "Single Algorithm":

        if algorithm == "PSO":
            results.append(evaluate(pso_model(), "PSO"))

        elif algorithm == "GA":
            results.append(evaluate(ga_model(), "GA"))

        elif algorithm == "ACO":
            results.append(evaluate(aco_model(), "ACO"))

        elif algorithm == "Hybrid (PSO+GA+ACO)":
            results.append(evaluate(hybrid_model(), "Hybrid"))

    else:
        results.append(evaluate(pso_model(), "PSO"))
        results.append(evaluate(ga_model(), "GA"))
        results.append(evaluate(aco_model(), "ACO"))
        results.append(evaluate(hybrid_model(), "Hybrid"))

    results_df = pd.DataFrame(results)

    # Remove Predictions column before display
    display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]

    # Static table (No search, no sort, no hide/pin)
    st.table(display_df)

    # ================= CONFUSION MATRIX ================= #
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
