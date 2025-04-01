import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

# Load sample regression dataset
def load_data():
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=5, noise=15, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple decision trees using bootstrapped samples
def train_multiple_trees(X_train, y_train, n_trees=10):
    trees = []
    for _ in range(n_trees):
        X_sample, y_sample = resample(X_train, y_train, random_state=np.random.randint(1000))
        tree = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree.fit(X_sample, y_sample)
        trees.append(tree)
    return trees

# Aggregate predictions from multiple trees
def aggregate_predictions(trees, X_test):
    predictions = np.array([tree.predict(X_test) for tree in trees])  # Shape (n_trees, n_samples)
    return np.mean(predictions, axis=0)  # Average predictions

# Main function
def main():
    X_train, X_test, y_train, y_test = load_data()
    trees = train_multiple_trees(X_train, y_train, n_trees=10)
    y_pred = aggregate_predictions(trees, X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

if __name__ == "__main__":
    main()
