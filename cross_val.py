import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def cross_validation(model, X, y, cv=10, scoring='r2', classification=False):
    """
    Perform cross-validation on the given model and return the best model based on
    minimal standard deviation of scores.

    Parameters:
    - model: Machine learning model (e.g., DecisionTreeRegressor or DecisionTreeClassifier)
    - X: Feature matrix
    - y: Target variable
    - cv: Number of folds (default: 5)
    - scoring: Metric to evaluate model ('r2' for regression, 'accuracy' for classification)
    - classification: Boolean flag to switch between KFold and StratifiedKFold

    Returns:
    - best_model: Model with the lowest standard deviation of scores
    """

    if classification:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=22)
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=22)

    models = []
    fold_scores = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # model_clone = type(model)()  # Create a fresh instance of the same model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if scoring == 'r2' and not classification:
            score = r2_score(y_test, y_pred)
        elif scoring == 'accuracy' and classification:
            score = accuracy_score(y_test, y_pred)
        else:
            raise ValueError("Unsupported scoring method. Use 'r2' for regression or 'accuracy' for classification.")

        models.append(model)
        fold_scores.append(score)

    fold_scores = np.array(fold_scores)
    mean=np.mean(fold_scores)
    # Compute the standard deviation for each fold
    sd = np.std(fold_scores)
    return mean,sd,models[1]

if __name__=="__main__":
    cross_validation()