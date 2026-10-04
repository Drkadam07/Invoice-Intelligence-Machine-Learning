from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def train_random_forest_model(x_train, y_train):
    """Train a Random Forest Classifier model and return the trained model."""
    ef= RandomForestClassifier( max_depth=5, random_state=42)

    parameter_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ['gini', 'entropy']
    }

    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(estimator=ef, param_grid=parameter_grid, scoring=scorer, cv=5,verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search

def evaluate_model(model, x_test, y_test):
    """Evaluate the model and return the metrics."""
    y_pred = model.predict(x_test)

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    return {
        "f1_score": f1,
        "roc_auc_score": roc_auc,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }