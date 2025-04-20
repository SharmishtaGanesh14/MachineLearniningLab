# Date created: 20th April 2025
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def load_data():
    data = pd.read_csv("../../datasets/Iris.csv")
    X = data.loc[:, ["SepalLengthCm", "SepalWidthCm"]]
    y = data["Species"]

    # Add noise and bin
    noise = np.random.normal(0, 0.1, size=X.shape)
    X_noisy = X + noise
    X_noisy["SepalLengthCm"] = pd.cut(X["SepalLengthCm"], bins=3, labels=[0, 1, 2])
    X_noisy["SepalWidthCm"] = pd.cut(X["SepalWidthCm"], bins=3, labels=[0, 1, 2])

    return X_noisy.astype(int), y


class JointProb:
    def __init__(self, X_train, y_train):
        self.X_train = X_train.reset_index(drop=True)
        self.y_train = y_train
        self.Table = []
        self.classes = np.unique(self.y_train)
        self.counts = Counter(self.y_train)
        self.probvalues = []

    def fit(self):
        X1 = np.unique(self.X_train.iloc[:, 0])
        X2 = np.unique(self.X_train.iloc[:, 1])
        for x in X1:
            for y in X2:
                self.Table.append([x, y])
        for row in self.Table:
            val = []
            for c in self.classes:
                count = sum(1 for i in range(len(self.X_train))
                            if list(self.X_train.iloc[i]) == row and self.y_train[i] == c)
                prob = count / self.counts[c] if self.counts[c] != 0 else 0
                val.append(prob)
            self.probvalues.append(val)

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            row = list(X_test.iloc[i])
            if row in self.Table:
                idx = self.Table.index(row)
                pred_class = np.argmax(self.probvalues[idx])
                predictions.append(pred_class)
            else:
                predictions.append(np.random.choice(len(self.classes)))  # handle unseen case
        return predictions


def evaluation(model, X_train, X_test, y_train, y_test, l):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Evaluation metrics:\n", classification_report(y_test, y_pred, target_names=l.classes_))


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=532, stratify=y)

    label = LabelEncoder()
    y_train_enc = label.fit_transform(y_train)
    y_test_enc = label.transform(y_test)

    # Model 1 - Decision tree classifier
    print(f"\n=== Decision Tree Classifier ===")
    model = DecisionTreeClassifier(max_depth=2)
    evaluation(model, X_train, X_test, y_train_enc, y_test_enc, label)

    # Model 2 - Simple Joint probability distribution
    print("\n=== Simple Joint Probability Distribution ===")
    jp_model = JointProb(X_train, y_train_enc)
    jp_model.fit()
    y_pred = jp_model.predict(X_test)
    print("Evaluation metrics:\n", classification_report(y_test_enc, y_pred, target_names=label.classes_))


if __name__ == "__main__":
    main()
