import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import numpy as np

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X,y

def train_classification_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc}")

class Node:
    def __init__(self,feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Class label for leaf nodes

def decision_tree_classifier_from_scratch(X,X_t,y,y_t):
    def entropy(labels):
        total_samples = len(labels)
        label_counts = Counter(labels)
        entropy_value = 0
        for count in label_counts.values():
            prob = count / total_samples
            entropy_value -= prob * np.log2(prob)
        return entropy_value

    def information_gain(parent_labels, left_child_labels, right_child_labels):
        total_parent = len(parent_labels)
        total_left = len(left_child_labels)
        total_right = len(right_child_labels)

        # Calculate entropies
        parent_entropy = entropy(parent_labels)
        left_entropy = entropy(left_child_labels)
        right_entropy = entropy(right_child_labels)

        # Weighted entropy of children
        weighted_child_entropy = (total_left / total_parent) * left_entropy + (
                    total_right / total_parent) * right_entropy

        # Information Gain
        info_gain = parent_entropy - weighted_child_entropy
        return info_gain

    def best_split(X,y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None
        parent_entropy = entropy(y)
        best_info_gain = 0
        best_feature = None
        best_threshold = None
        for feature in range(num_features):
            thresholds=np.unique(X[:, feature])
            for threshold in thresholds:
                x_left=X[:,feature]<=threshold
                x_right=X[:,feature]>threshold

                if sum(x_left) == 0 or sum(x_right) == 0:
                    continue

                info_gain =information_gain(y,y[x_left],y[x_right])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


    def BuildTree(X,y,depth=0,max_depth=10):
        if len(set(y)) == 1:
            return Node(value=y[0])
        if depth>=max_depth:
            return Node(value=Counter(y).most_common(1)[0][0])
        feature,threshold=best_split(X,y)
        if feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])
        left_mask=X[:,feature] <= threshold
        right_mask=X[:,feature] > threshold
        left_child=BuildTree(X[left_mask],y[left_mask],depth+1,max_depth)
        right_child = BuildTree(X[right_mask],y[right_mask], depth + 1, max_depth)
        return Node(feature=feature,threshold=threshold,left=left_child,right=right_child)

    def predict_one(node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return predict_one(node.left, x)
        return predict_one(node.right, x)

    def predict(tree, X):
        return np.array([predict_one(tree, x) for x in X])

    tree = BuildTree(X=X, y=y)

    # Make predictions
    y_pred = predict(tree, X_t)

    # Evaluate model
    accuracy = accuracy_score(y_t, y_pred)
    print(f"Accuracy score: {accuracy:.4f}")

def main():
    # Exercise 1
    # parent_labels = ['A', 'A', 'B', 'B', 'B']
    # left_child_labels = ['A', 'B']
    # right_child_labels = ['A', 'B', 'B']
    #
    # print("Entropy of parent:", entropy(parent_labels))
    # print("Information Gain:", information_gain(parent_labels, left_child_labels, right_child_labels))

    X1,y1=load_data()
    X_Train1, X_Test1, y_Train1, y_Test1 = train_test_split(X1, y1, test_size=0.30, random_state=999)

    print("\nAccuracy value for Decision tree classification [SCI-KIT]:")
    train_classification_tree(X_Train1, X_Test1, y_Train1, y_Test1)

    print("\nAccuracy value for Decision tree classification [FROM SCRATCH]:")
    decision_tree_classifier_from_scratch(X_Train1, X_Test1, y_Train1, y_Test1)


if __name__=="__main__":
    main()