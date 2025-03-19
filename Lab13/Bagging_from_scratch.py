from collections import Counter

import numpy as np
import random

from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

from Lab12.Ex12 import load_data, Decision_tree_regression_fromscratch, predict
from Lab10and11.Ex10_11 import load_data2, decision_tree_classifier_from_scratch
from Lab13.Bagging_SciKit import Data_processing_reg, Data_processing_class


def Bagging(X, y, bags=10, task='regression'):
    models = []

    def Bootstrap(X, y):
        n = X.shape[0]
        random.seed(1)
        indices = [random.randint(0, n - 1) for _ in range(n)]
        return X[indices, :], y[indices]

    for _ in range(bags):
        X_boot, y_boot = Bootstrap(X, y)
        if task == 'regression':
            model = Decision_tree_regression_fromscratch(X_boot, y_boot, max_depth=5, min_samples_split=5)
        elif task == 'classification':
            model = decision_tree_classifier_from_scratch(X_boot, y_boot, max_depth=5, min_samples_split=5)
        else:
            raise ValueError("Task must be either 'regression' or 'classification'")

        models.append(model)

    return models


def aggregate(X_test, models, task='regression'):
    y_preds = []

    for model in models:
        y_pred = predict(model, X_test)
        y_preds.append(y_pred)

    if task == 'regression':
        return np.mean(y_preds, axis=0)
    elif task == 'classification':
        # For classification, apply majority voting (mode)
        # Transpose to get predictions per sample
        y_preds = np.array(y_preds).T  # Shape becomes (num_test_samples, num_trees)
        # Apply majority voting
        Y = []
        for sample_preds in y_preds:
            y_counts = Counter(sample_preds)
            majority_class = max(y_counts, key=y_counts.get)  # Most frequent class
            Y.append(majority_class)
        return np.array(Y)

def main():
    # Regression
    # X, y = load_data()
    from sklearn.datasets import load_diabetes
    import pandas as pd
    data=load_diabetes()
    X=pd.DataFrame(data.data)
    y=pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=222)
    X_train, X_test, y_train, y_test = Data_processing_reg(X_train, X_test, y_train, y_test)

    # Bagging for Regression
    models_reg = Bagging(X_train, y_train, bags=10, task='regression')
    y_pred_reg = aggregate(X_test, models_reg, task='regression')
    r2 = r2_score(y_test, y_pred_reg)
    print(f"R² Score (Regression): {r2:.4f}")

    # Classification
    # X_class, y_class = load_data2()
    import pandas as pd
    data=load_iris()
    X_class=pd.DataFrame(data.data)
    y_class=pd.Series(data.target)
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.30,
                                                                                random_state=222)
    X_train_class, X_test_class, y_train_class, y_test_class = Data_processing_class(X_train_class, X_test_class, y_train_class, y_test_class)

    # Bagging for Classification
    models_class = Bagging(X_train_class, y_train_class, bags=10, task='classification')
    y_pred_class = aggregate(X_test_class, models_class, task='classification')
    acc = accuracy_score(y_test_class, y_pred_class)
    print(f"Accuracy Score (Classification): {acc:.4f}")


if __name__ == "__main__":
    main()
