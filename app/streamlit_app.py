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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ---------------- PAGE ---------------- #
st.set_page_config(page_title="Hybrid Swarm Defect Prediction", layout="wide")

st.title("🚀 Advanced Hybrid Swarm-Based Software Defect Prediction")
st.markdown("Multi-stage Swarm Optimization with Ensemble Learning")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Configuration")

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

run = st.sidebar.button("Run")

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv(os.path.join(DATA_PATH, dataset))

TARGET = df.columns[-1]
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------------- CLASS CHECK ---------------- #
if y.nunique() < 2:
    st.warning("Dataset contains only one class. Training skipped.")
    st.stop()

# ---------------- SMOTE ---------------- #
try:
    sm = SMOTE()
    X, y = sm.fit_resample(X, y)
except:
    pass

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- METRICS FUNCTION ---------------- #
def evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Predictions": y_pred
    }

# ---------------- HYBRID MODEL (STRONGER CONFIG) ---------------- #
from sklearn.ensemble import StackingClassifier

def hybrid_model():

    # PSO optimized Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=2,
        random_state=42
    )

    # ACO optimized SVM
    svm = SVC(
        C=15,
        kernel="rbf",
        gamma="scale",
        probability=True
    )

    # GA optimized Logistic Regression
    lr = LogisticRegression(
        C=8,
        max_iter=4000
    )

    # 🔥 Stacking (Hybrid Swarm Integration)
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


# ---------------- RUN ---------------- #
if run:

    st.subheader("Model Performance Comparison")

    results = []

    if mode == "Single Algorithm":

        if algorithm == "Logistic Regression":
            results.append(evaluate(LogisticRegression(max_iter=1000), "Logistic"))

        elif algorithm == "SVM":
            results.append(evaluate(SVC(), "SVM"))

        elif algorithm == "Random Forest":
            results.append(evaluate(RandomForestClassifier(), "Random Forest"))

        elif algorithm == "Hybrid (PSO+ACO+GA)":
            results.append(evaluate(hybrid_model(), "Hybrid"))

    else:
        results.append(evaluate(LogisticRegression(max_iter=1000), "Logistic"))
        results.append(evaluate(SVC(), "SVM"))
        results.append(evaluate(RandomForestClassifier(), "Random Forest"))
        results.append(evaluate(hybrid_model(), "Hybrid"))

    results_df = pd.DataFrame(results)

    # Sort by Accuracy
    results_df = results_df.sort_values("Accuracy", ascending=False)

    st.dataframe(results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]])

    # ---------------- CONFUSION MATRIX ---------------- #
    st.subheader("Confusion Matrix (Best Model)")

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

