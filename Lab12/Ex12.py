# CODE WRITTEN ON: 04/02/2025
# Implement a decision regression tree algorithm without using scikit-learn using the diabetes dataset. Fetch the dataset from scikit-learn library.

import pandas as pd
import numpy as np
from numpy.ma.extras import unique
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# Define Node class for the Decision Tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Index of the feature used for splitting
        self.threshold = threshold  # Threshold value for the split
        self.right = right  # Right child node (greater than threshold)
        self.left = left  # Left child node (less than or equal to threshold)
        self.value = value  # Predicted value (for leaf nodes)


# Load data function
def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")  # Load dataset
    X = data.drop(columns=["disease_score", "disease_score_fluct"], axis=1).values  # Features
    y = data["disease_score"].values  # Target variable
    return X, y


# Function to find the best split
def best_split(X, y, min_samples_split=2):
    num_samples, num_features = X.shape  # Get number of samples and features
    Error_prev = float('inf')  # Initialize best error to a large value
    best_feature = None  # Best feature index for splitting
    best_threshold = None  # Best threshold value for splitting

    for i in range(num_features):  # Iterate over all features
        thresholds = unique(X[:, i])  # Unique values in feature column

        for j in thresholds:  # Iterate over all unique threshold values
            left_mask = X[:, i] <= j  # Identify left split (<= threshold)
            right_mask = X[:, i] > j  # Identify right split (> threshold)

            # Skip if one side is empty
            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue

            y_left, y_right = y[left_mask], y[right_mask]  # Split target variable accordingly
            y_label_left = np.mean(y_left)  # Mean of left split
            y_label_right = np.mean(y_right)  # Mean of right split

            # Calculate squared error
            Error = np.sum((y_left - y_label_left) ** 2) + np.sum((y_right - y_label_right) ** 2)

            # Update best split if current split has lower error
            if Error < Error_prev:
                Error_prev = Error
                best_feature = i
                best_threshold = j

    return best_feature, best_threshold  # Return best feature index and threshold


# Function to build the decision tree
def BuildTree(X, y, depth=0, max_depth=10, min_samples_split=2):
    # If max depth reached or sample size is too small, return a leaf node
    if depth >= max_depth or len(y) < min_samples_split:
        return Node(value=np.mean(y))  # Return leaf node with mean target value

    feature, threshold = best_split(X, y)  # Find the best feature and threshold

    # if feature is None or threshold is None:
    #     return Node(value=np.mean(y))  # Return leaf if no split found

    left_mask = X[:, feature] <= threshold  # Mask for left subtree
    right_mask = X[:, feature] > threshold  # Mask for right subtree

    # Recursively build the left and right subtrees
    left = BuildTree(X[left_mask], y[left_mask], depth=depth + 1, max_depth=max_depth)
    right = BuildTree(X[right_mask], y[right_mask], depth=depth + 1, max_depth=max_depth)

    return Node(feature=feature, threshold=threshold, left=left, right=right)  # Return a decision node


# Prediction function for a single sample
def predict_one(node, x):
    if node.value is not None:  # If leaf node, return stored value
        return node.value
    if x[node.feature] <= node.threshold:  # Traverse left if condition is met
        return predict_one(node.left, x)
    return predict_one(node.right, x)  # Otherwise, traverse right


# Prediction function for multiple samples
def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])  # Apply predict_one() to each sample

def main():
    # Load dataset
    X, y = load_data()

    # Split into train and test sets
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=22)
    # X_Train=np.array([[1,2],[3,6]])  # Example training data
    # y_Train=np.array([1,5])  # Example labels for training data
    # X_Test=np.array([[5,1]])  # Example test data
    # y_Test=np.array([2])  # Example labels for test data

    # Build the decision tree
    tree = BuildTree(X_Train, y_Train, max_depth=10)

    # Make predictions
    y_pred = predict(tree, X_Test)

    # Compute R2 Score
    r2Score = r2_score(y_Test, y_pred)
    print(f"R2 Score: {r2Score:.2f}")  # Print R2 score rounded to two decimal places

if __name__=="__main__":
    main()