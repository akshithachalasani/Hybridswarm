from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def optimize_model(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    model = RandomForestClassifier(random_state=42)

    search = RandomizedSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='f1',
        n_iter=5
    )

    search.fit(X, y)
    return search.best_estimator_
