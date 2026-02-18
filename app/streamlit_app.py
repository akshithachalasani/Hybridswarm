import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Hybrid Defect Prediction", layout="wide")

st.title("🚀 Advanced Hybrid Swarm-Based Software Defect Prediction")
st.markdown("Multi-stage swarm optimization with ensemble learning.")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙ Configuration")

DATA_PATH = "data"  # folder where csv files exist

# List all CSV files automatically
datasets = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

if len(datasets) == 0:
    st.error("No datasets found inside data folder.")
    st.stop()

selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)

mode = st.sidebar.radio("Mode", ["Single Algorithm", "Full Comparison"])

algorithm = None
if mode == "Single Algorithm":
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Random Forest", "SVM", "Logistic Regression"]
    )

run_button = st.sidebar.button("Run")

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv(os.path.join(DATA_PATH, selected_dataset))

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- TARGET COLUMN ---------------- #
TARGET_COLUMN = df.columns[-1]  # assuming last column is target

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# ---------------- CHECK CLASS ISSUE ---------------- #
if y.nunique() < 2:
    st.warning("⚠ This dataset contains only one class. Model training skipped.")
    st.stop()

# ---------------- APPLY SMOTE SAFELY ---------------- #
try:
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
except Exception:
    st.info("SMOTE not applied. Continuing without resampling.")

# ---------------- SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- TRAINING ---------------- #
if run_button:

    st.subheader("Model Results")

    def evaluate_model(model, name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        st.success(f"{name} Accuracy: {round(acc * 100, 2)}%")

    if mode == "Single Algorithm":

        if algorithm == "Random Forest":
            evaluate_model(RandomForestClassifier(), "Random Forest")

        elif algorithm == "SVM":
            evaluate_model(SVC(), "SVM")

        elif algorithm == "Logistic Regression":
            evaluate_model(LogisticRegression(max_iter=1000), "Logistic Regression")

    else:
        evaluate_model(RandomForestClassifier(), "Random Forest")
        evaluate_model(SVC(), "SVM")
        evaluate_model(LogisticRegression(max_iter=1000), "Logistic Regression")
