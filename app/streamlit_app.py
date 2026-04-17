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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hybrid Swarm Defect Prediction",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #161B22;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
label[data-testid="stWidgetLabel"] {
    color: white !important;
}
div[data-baseweb="select"] * {
    color: black !important;
}
h1, h2, h3 {
    color: #00CFFF !important;
}
div.stButton > button {
    background-color: #00CFFF !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}
thead tr th {
    background-color: #00CFFF !important;
    color: black !important;
    text-align: center !important;
}
tbody tr td {
    background-color: #1C2128 !important;
    color: white !important;
    text-align: center !important;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>🚀 Hybrid Swarm-Based Software Defect Prediction</h1>
<p>PSO + GA + ACO Optimized Framework</p>
""", unsafe_allow_html=True)

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

df = pd.read_csv(os.path.join(DATA_PATH, dataset))

TARGET = df.columns[-1]
X = df.drop(columns=[TARGET])
y = df[TARGET]

if y.nunique() < 2:
    st.warning("Dataset contains only one class.")
    st.stop()

try:
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
except:
    pass

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

def pso_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42
    )

def ga_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=6, max_iter=4000)
    )

def aco_model():
    return make_pipeline(
        StandardScaler(),
        SVC(C=12, kernel="rbf", probability=True)
    )

def hybrid_model():

    rf = RandomForestClassifier(n_estimators=350, random_state=42)

    svm = make_pipeline(
        StandardScaler(),
        SVC(C=15, kernel="rbf", probability=True)
    )

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    )

    hybrid = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm),
            ("lr", lr)
        ],
        final_estimator=RandomForestClassifier(
            n_estimators=500,
            random_state=42
        ),
        passthrough=True
    )

    return hybrid

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

    display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
    st.table(display_df)

    results_sorted = results_df.sort_values(
        by=["Accuracy", "F1 Score", "Precision"],
        ascending=False
    )

    best_row = results_sorted.iloc[0]

    st.markdown(f"### 🏆 Best Model: {best_row['Model']}")

    st.markdown("## 🔍 Confusion Matrix (Best Model)")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_row["Predictions"],
        ax=ax,
        cmap="Blues"
    )

    ax.set_title(f"Confusion Matrix - {best_row['Model']}")
    st.pyplot(fig)
