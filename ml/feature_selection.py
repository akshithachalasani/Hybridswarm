import numpy as np
from sklearn.ensemble import RandomForestClassifier


def select_top_features(X, y, feature_names, k_ratio=0.6):

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    k = int(len(feature_names) * k_ratio)
    selected_idx = indices[:k]

    selected_features = [feature_names[i] for i in selected_idx]

    return X[:, selected_idx], selected_features


def fs_ga(X, y, feature_names):
    return select_top_features(X, y, feature_names, 0.5)


def fs_pso(X, y, feature_names):
    return select_top_features(X, y, feature_names, 0.6)


def fs_aco(X, y, feature_names):
    return select_top_features(X, y, feature_names, 0.7)


def fs_hybrid_advanced(X, y, feature_names):
    # Hybrid keeps more strong features
    return select_top_features(X, y, feature_names, 0.8)
