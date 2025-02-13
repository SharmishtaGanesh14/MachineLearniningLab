import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline


def load_data():
    data = pd.read_csv("Sonar.csv", header=None)
    column_names = [f"Frequency{i}" for i in range(1, data.shape[1])] + ["Class"]
    data.columns = column_names
    y = data["Class"].map({'M': 1, 'R': 0})  # Map target variable to binary
    X = data.drop("Class", axis=1)
    return X, y


def perform_eda():
    X, y = load_data()

    print("\n=== Dataset Description ===")
    print(X.describe())
    print("\n=== Data Information ===")
    print(X.info())
    print("\n=== Checking for Null Values ===")
    print(X.isnull().sum())

    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=False, cmap="coolwarm", cbar=True)
    plt.title("Feature Correlation Matrix")
    plt.show()

    y.value_counts().plot(kind='bar', color=["blue", "orange"], edgecolor="black")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.xticks(ticks=[0, 1], labels=["Rock (0)", "Metal (1)"])
    plt.show()


def standardize_manual(X, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


def logistic_regression_k_folds():
    X, y = load_data()
    model = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 1. Logistic Regression WITHOUT Preprocessing
    scores_raw = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Accuracy without Preprocessing: {np.mean(scores_raw):.4f} ± {np.std(scores_raw):.4f}")

    # 2. Logistic Regression with MANUAL Standardization
    accuracies = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Apply standardization only on training data
        X_train_scaled, mean, std = standardize_manual(X_train)
        X_test_scaled, _, _ = standardize_manual(X_test, mean, std)  # Use training mean/std

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred))

    print(f"Accuracy (Manual Scaling): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

    # 3. Logistic Regression with SCIKIT-LEARN StandardScaler
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(max_iter=1000))
    ])
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"Accuracy (Scikit-Learn Scaling): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


if __name__ == "__main__":
    # perform_eda()  # Uncomment to run EDA
    logistic_regression_k_folds()
