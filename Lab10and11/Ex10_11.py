# CODE WRITTEN ON: 03/02/2025
# Implement decision tree classifier without using scikit-learn using the iris dataset. Fetch the iris dataset from scikit-learn library.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import numpy as np


# Function to load the Iris dataset
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


# Function to train a Decision Tree Classifier using Sci-Kit Learn
def train_classification_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Predict on test set
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f"Accuracy score: {acc:.4f}")  # Print accuracy score with 4 decimal places


# Node class for the Decision Tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index used for splitting
        self.threshold = threshold  # Split threshold
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class label for leaf nodes


# Function to build a Decision Tree Classifier from scratch
def decision_tree_classifier_from_scratch(X, X_t, y, y_t):
    # Function to compute entropy
    def entropy(labels):
        total_samples = len(labels)
        label_counts = Counter(labels)  # Count occurrences of each class label
        entropy_value = 0
        for count in label_counts.values():
            prob = count / total_samples
            entropy_value -= prob * np.log2(prob)  # Compute entropy formula
        return entropy_value

    # Function to compute information gain
    def information_gain(parent_labels, left_child_labels, right_child_labels):
        total_parent = len(parent_labels)
        total_left = len(left_child_labels)
        total_right = len(right_child_labels)

        # Calculate entropy of parent and children
        parent_entropy = entropy(parent_labels)
        left_entropy = entropy(left_child_labels)
        right_entropy = entropy(right_child_labels)

        # Weighted entropy of children
        weighted_child_entropy = (total_left / total_parent) * left_entropy + (
                    total_right / total_parent) * right_entropy

        # Compute Information Gain
        info_gain = parent_entropy - weighted_child_entropy
        return info_gain

    # Function to find the best feature and threshold for splitting
    def best_split(X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None

        best_info_gain = 0
        best_feature = None
        best_threshold = None

        for feature in range(num_features):  # Iterate over all features
            thresholds = np.unique(X[:, feature])  # Get unique threshold values

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold  # Mask for left split
                right_mask = X[:, feature] > threshold  # Mask for right split

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue  # Skip splits with empty partitions

                info_gain = information_gain(y, y[left_mask], y[right_mask])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # Recursive function to build the tree
    def BuildTree(X, y, depth=0, max_depth=10):
        if len(set(y)) == 1:  # If all labels are the same, return leaf node
            return Node(value=y[0])

        if depth >= max_depth:  # If max depth reached, return most common label
            return Node(value=Counter(y).most_common(1)[0][0])

        feature, threshold = best_split(X, y)
        if feature is None:  # If no valid split found, return majority label
            return Node(value=Counter(y).most_common(1)[0][0])

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        # Recursively build left and right subtrees
        left_child = BuildTree(X[left_mask], y[left_mask], depth + 1, max_depth)
        right_child = BuildTree(X[right_mask], y[right_mask], depth + 1, max_depth)

        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    # Function to predict a single sample
    def predict_one(node, x):
        if node.value is not None:  # If leaf node, return stored value
            return node.value
        if x[node.feature] <= node.threshold:  # Traverse left if condition is met
            return predict_one(node.left, x)
        return predict_one(node.right, x)  # Otherwise, traverse right

    # Function to predict multiple samples
    def predict(tree, X):
        return np.array([predict_one(tree, x) for x in X])  # Apply predict_one() to each sample

    # Build the decision tree
    tree = BuildTree(X=X, y=y)

    # Make predictions
    y_pred = predict(tree, X_t)

    # Evaluate model accuracy
    accuracy = accuracy_score(y_t, y_pred)
    print(f"Accuracy score: {accuracy:.4f}")  # Print accuracy score with 4 decimal places


# Main function
def main():
    # Uncomment this block if you want to test entropy and information gain calculations
    # parent_labels = ['A', 'A', 'B', 'B', 'B']
    # left_child_labels = ['A', 'B']
    # right_child_labels = ['A', 'B', 'B']
    # print("Entropy of parent:", entropy(parent_labels))
    # print("Information Gain:", information_gain(parent_labels, left_child_labels, right_child_labels))

    # Load dataset
    X1, y1 = load_data()

    # Split data into training and testing sets
    X_Train1, X_Test1, y_Train1, y_Test1 = train_test_split(X1, y1, test_size=0.30, random_state=999)

    # Train Decision Tree using Sci-Kit Learn
    print("\nAccuracy value for Decision tree classification [SCI-KIT]:")
    train_classification_tree(X_Train1, X_Test1, y_Train1, y_Test1)

    # Train Decision Tree from Scratch
    print("\nAccuracy value for Decision tree classification [FROM SCRATCH]:")
    decision_tree_classifier_from_scratch(X_Train1, X_Test1, y_Train1, y_Test1)


# Run main function
if __name__ == "__main__":
    main()
