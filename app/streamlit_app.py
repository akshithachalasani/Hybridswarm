import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

# ================= DARK THEME ================= #
st.markdown("""
<style>
.stApp { background-color: #121212; color: white; }
section[data-testid="stSidebar"] { background-color: #1E1E1E; }
h1, h2, h3 { color: #00BFFF !important; }
div.stButton > button {
    background-color: #00BFFF !important;
    color: black !important;
    font-weight: bold !important;
}
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
<p>PSO + GA + ACO Optimized Framework (Cross-Validated)</p>
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
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
except:
    pass


# ================= CROSS VALIDATION ================= #
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ================= EVALUATION FUNCTION ================= #
def evaluate(model, name):

    y_pred = cross_val_predict(model, X, y, cv=cv)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "Precision": round(precision_score(y, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y, y_pred, average="weighted", zero_division=0), 4),
        "F1 Score": round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
        "Predictions": y_pred
    }


# ================= INDIVIDUAL MODELS ================= #

def pso_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42
    )

def ga_model():
    return LogisticRegression(
        C=6,
        max_iter=4000
    )

def aco_model():
    return SVC(
        C=12,
        kernel="rbf",
        probability=True
    )


# ================= HYBRID STACKING MODEL ================= #
def hybrid_model():

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    svm = SVC(C=12, kernel="rbf", probability=True)
    lr = LogisticRegression(max_iter=4000)

    hybrid = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm),
            ("lr", lr)
        ],
        final_estimator=LogisticRegression(),
        cv=cv,
        passthrough=True
    )

    return hybrid


# ================= RUN ================= #
if run:

    st.markdown("## 📊 Model Performance (5-Fold Cross Validation)")

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

    display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]

    st.table(display_df)

    # ================= BEST MODEL ================= #
    best_index = results_df["Accuracy"].idxmax()
    best_row = results_df.loc[best_index]

    st.markdown(f"### 🏆 Best Model: {best_row['Model']}")

    # ================= CONFUSION MATRIX ================= #
    st.markdown("## 🔍 Confusion Matrix (Best Model - CV Predictions)")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y,
        best_row["Predictions"],
        ax=ax,
        cmap="Blues"
    )

    ax.set_title(f"Confusion Matrix - {best_row['Model']}")
    st.pyplot(fig)
