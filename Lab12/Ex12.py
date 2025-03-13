# CODE WRITTEN ON: 04/02/2025
# Implement a decision regression tree algorithm without using scikit-learn using the diabetes dataset. Fetch the dataset from scikit-learn library.
# CODE MODIFIED ON: 11/03/2025

import pandas as pd
import numpy as np
from numpy.ma.extras import unique
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from cross_val import cross_validation

# Load data function
def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")  # Load dataset
    X = data.drop(columns=["disease_score", "disease_score_fluct"], axis=1).values  # Features
    y = data["disease_score"].values  # Target variable
    return X, y

def load_data2():
    data = pd.read_csv("../Lab9/breast_cancer.csv")
    X = data.drop(columns=["diagnosis", "id", "Unnamed: 32"]).values
    data["diagnosis_binary"] = data["diagnosis"].map({'M': 1, 'B': 0})
    y = data["diagnosis_binary"].values
    return X, y

def Decision_tree_regression_Scikit(X_Train, X_Test, y_Train, y_Test,max_depth,min_samples_split):
    # Create and train a Decision Tree Regressor
    regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    regressor.fit(X_Train, y_Train)

    # Make predictions on the test set
    y_pred = regressor.predict(X_Test)

    # Evaluate model performance
    r2 = r2_score(y_Test, y_pred)
    return r2

# Define Node class for the Decision Tree
def Decision_tree_regression_fromscratch(X_Train,y_Train,max_depth=5,min_samples_split=5):
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature  # Index of the feature used for splitting
            self.threshold = threshold  # Threshold value for the split
            self.right = right  # Right child node (greater than threshold)
            self.left = left  # Left child node (less than or equal to threshold)
            self.value = value  # Predicted value (for leaf nodes)

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
    def BuildTree(X, y, depth=0):
        # If max depth reached or sample size is too small, return a leaf node
        if depth >= max_depth or len(y) < min_samples_split:
            return Node(value=np.mean(y))  # Return leaf node with mean target value

        feature, threshold = best_split(X, y)  # Find the best feature and threshold

        if feature is None or threshold is None:
            return Node(value=np.mean(y))  # Return leaf if no split found

        left_mask = X[:, feature] <= threshold  # Mask for left subtree
        right_mask = X[:, feature] > threshold  # Mask for right subtree

        # Recursively build the left and right subtrees
        left = BuildTree(X[left_mask], y[left_mask], depth=depth + 1)
        right = BuildTree(X[right_mask], y[right_mask], depth=depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)  # Return a decision node

    tree = BuildTree(X_Train, y_Train)
    return tree

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

def hyperparameter_tuning(l,a,X,X_t,y,y_t):
    R2 = []
    hyper_params = {}

    for val1 in l:  # max_depth values
        for val2 in a:  # min_samples_split values
            r2 = Decision_tree_regression_Scikit(X, X_t, y, y_t, max_depth=val1, min_samples_split=val2)
            R2.append(r2)
            hyper_params[(val1, val2)] = r2  # Store params as a tuple

    R2 = np.array(R2)
    max_ind = np.argmax(R2)  # Index of the best R² score
    best_params = list(hyper_params.keys())[max_ind]  # Extract best params from dictionary

    return best_params[0],best_params[1]


def main():
    # Load dataset
    X, y = load_data2()

    # WITHOUT CROSS VALIDATION
    # Split into train and test sets
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=42)
    # X_Train=np.array([[1,2],[3,6]])  # Example training data
    # y_Train=np.array([1,5])  # Example labels for training data
    # X_Test=np.array([[5,1]])  # Example test data
    # y_Test=np.array([2])  # Example labels for test data
    tree=Decision_tree_regression_fromscratch(X_Train, y_Train)
    # Make predictions
    y_pred = predict(tree, X_Test)
    # Compute R2 Score
    r2Score = r2_score(y_Test, y_pred)
    print(f"R² Score: {r2Score:.2f}")  # Print R2 score rounded to two decimal places

    print(f"R² Score: {Decision_tree_regression_Scikit(X_Train, X_Test, y_Train, y_Test,max_depth=15, min_samples_split=10)}")


    #WITH CROSS VALIDATION AND HYPER PARAMETER TUNING-SCIKIT
    lamb=[2,5,10,15]
    alpha=[2,5,10,15]

    max_depth,min=hyperparameter_tuning(lamb,alpha,X_Train,X_Test,y_Train,y_Test)
    res1,res2,res3=cross_validation(DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min, random_state=42),X,y,scoring='r2',classification=False)
    print(f"R2 score mean: {res1}")
    print(f"R2 standard dev: {res2}")

if __name__=="__main__":
    main()