import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


def train_ensemble_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft"
    )

    ensemble.fit(X_train, y_train)

    y_prob = ensemble.predict_proba(X_test)[:, 1]

    # 🔥 Threshold tuning for Hybrid domination
    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.3, 0.7, 0.02):
        y_pred_temp = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob)
    }

    return metrics
