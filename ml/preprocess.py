import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def preprocess_data(path):

    df = pd.read_csv(path)

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # -----------------------------
    # SAFE TARGET CLEANING (NO APPLY)
    # -----------------------------

    # Convert to numeric safely
    y = pd.to_numeric(y, errors="coerce")

    # Replace NaN with 0
    y = y.fillna(0)

    # Convert to binary
    y = np.where(y > 0, 1, 0)

    y = pd.Series(y)

    # -----------------------------
    # CLEAN FEATURES
    # -----------------------------

    # Drop all-NaN columns
    X = X.dropna(axis=1, how="all")

    # Fill remaining NaN
    X = X.fillna(0)

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Apply SMOTE safely
    # -----------------------------

    if len(np.unique(y)) > 1:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_scaled, y)
    else:
        X_res, y_res = X_scaled, y

    return X_res, y_res, feature_names
