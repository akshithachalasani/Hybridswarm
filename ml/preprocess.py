import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import streamlit as st


def preprocess_data(df, target_column="bug"):
    """
    Preprocess dataset safely for training.
    Handles single-class datasets without crashing.
    """

    # Check if target column exists
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataset.")
        return None, None, None, None

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 🚨 FIX 1 — Handle single-class dataset
    if y.nunique() < 2:
        st.warning("⚠ Dataset contains only one class. Skipping model training.")
        return None, None, None, None

    # 🚨 FIX 2 — Apply SMOTE safely
    try:
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)
    except Exception as e:
        st.warning("SMOTE could not be applied. Continuing without resampling.")
        print("SMOTE error:", e)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
