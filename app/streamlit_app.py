import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np

from ml.preprocess import preprocess_data
from ml.feature_selection import fs_ga, fs_pso, fs_aco, fs_hybrid_advanced
from ml.models import train_ensemble_model


st.set_page_config(page_title="Hybrid Defect Prediction", layout="wide")

st.title("🚀 Advanced Hybrid Swarm-Based Software Defect Prediction")
st.write("Multi-stage swarm optimization with ensemble learning.")


# Sidebar
st.sidebar.header("⚙ Configuration")

dataset_folder = "data"
dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]

dataset_choice = st.sidebar.selectbox("Select Dataset", dataset_files)
mode = st.sidebar.radio("Mode", ["Single Algorithm", "Full Comparison"])
algorithm = st.sidebar.selectbox("Select Algorithm", ["GA", "PSO", "ACO", "Hybrid Advanced"])
run_btn = st.sidebar.button("Run")


# Load data
dataset_path = os.path.join(dataset_folder, dataset_choice)

X, y, feature_names = preprocess_data(dataset_path)

if len(np.unique(y)) < 2:
    st.error("Dataset has only one class.")
    st.stop()


def run_algorithm(algo):

    if algo == "GA":
        X_sel, selected = fs_ga(X, y, feature_names)
    elif algo == "PSO":
        X_sel, selected = fs_pso(X, y, feature_names)
    elif algo == "ACO":
        X_sel, selected = fs_aco(X, y, feature_names)
    else:
        X_sel, selected = fs_hybrid_advanced(X, y, feature_names)

    metrics = train_ensemble_model(X_sel, y)

    return metrics, selected


if run_btn:

    if mode == "Single Algorithm":

        metrics, selected = run_algorithm(algorithm)

        st.subheader(f"{algorithm} Performance")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Accuracy (%)", round(metrics["accuracy"] * 100, 2))
        col2.metric("Precision (%)", round(metrics["precision"] * 100, 2))
        col3.metric("Recall (%)", round(metrics["recall"] * 100, 2))
        col4.metric("F1 Score (%)", round(metrics["f1"] * 100, 2))
        col5.metric("AUC (%)", round(metrics["auc"] * 100, 2))

        st.write("Selected Features:", selected)

    else:

        results = {}

        for algo in ["GA", "PSO", "ACO", "Hybrid Advanced"]:
            metrics, _ = run_algorithm(algo)
            results[algo] = metrics

        comparison_df = pd.DataFrame({
            "Algorithm": results.keys(),
            "Accuracy (%)": [v["accuracy"] * 100 for v in results.values()],
            "Precision (%)": [v["precision"] * 100 for v in results.values()],
            "Recall (%)": [v["recall"] * 100 for v in results.values()],
            "F1 Score (%)": [v["f1"] * 100 for v in results.values()],
            "AUC (%)": [v["auc"] * 100 for v in results.values()],
        })

        st.subheader("Full Performance Comparison")
        st.dataframe(comparison_df)

        st.subheader("Accuracy Comparison")
        st.bar_chart(comparison_df.set_index("Algorithm")["Accuracy (%)"])
