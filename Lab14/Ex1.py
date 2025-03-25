import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load the Iris dataset
def load_data2():
    data = pd.read_csv("../Iris.csv")
    X = data.drop(columns=["Species", "Id"])
    y = data['Species']
    return X, y


# Preprocessing: Label encoding and scaling
def Data_processing(X_Train, X_Test, y_Train, y_Test):
    label = LabelEncoder()
    y_encoded_train = label.fit_transform(y_Train)
    y_encoded_test = label.transform(y_Test)

    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_Train)
    X_test_scaled = scaling.transform(X_Test)

    return X_train_scaled, X_test_scaled, y_encoded_train, y_encoded_test, label


# AdaBoost SAMME Classifier from Scratch
class AdaBoostSAMME:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # Unique class labels
        K = len(self.classes)  # Number of classes

        # Initialize weights uniformly
        w = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            # Train weak learner (Decision Stump)
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            # Compute weighted error
            incorrect = (y_pred != y).astype(int)
            err_t = np.dot(w, incorrect) / np.sum(w)

            # Avoid division by zero or trivial solutions
            err_t = max(err_t, 1e-10)
            if err_t >= 1 - 1e-10:  # If error is too high, stop boosting
                break

            # Compute alpha (weight of weak learner)
            alpha_t = np.log((1 - err_t) / err_t) + np.log(K - 1)

            # Update weights for next iteration (explicit condition)
            for i in range(len(w)):
                if incorrect[i] == 1:  # Misclassified
                    w[i] *= np.exp(alpha_t)  # Increase weight
                else:  # Correctly classified
                    w[i] *= np.exp(-alpha_t)  # Decrease weight

            # Normalize weights to maintain a probability distribution
            w /= np.sum(w)

            # Store model and alpha
            self.models.append(model)
            self.alphas.append(alpha_t)

    def predict(self, X):
        pred = np.zeros((X.shape[0], len(self.classes)))

        for alpha, model in zip(self.alphas, self.models):
            y_pred = model.predict(X)
            pred[np.arange(X.shape[0]), y_pred] += alpha  # Weighted vote

        return self.classes[np.argmax(pred, axis=1)]


# Main function to execute the pipeline
def main():
    # Load and split the Iris dataset
    X, y = load_data2()
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=4343)
    X_Train_scaled, X_Test_scaled, y_Train, y_Test, y_encoder = Data_processing(X_Train, X_Test, y_Train, y_Test)

    # Train and evaluate the scikit-learn AdaBoost model
    print("\nScikit-learn AdaBoost Classifier:")
    weak_learner = DecisionTreeClassifier(max_depth=1)
    adaboost = AdaBoostClassifier(estimator=weak_learner, n_estimators=50, random_state=42)
    adaboost.fit(X_Train_scaled, y_Train)
    y_pred_sklearn = adaboost.predict(X_Test_scaled)
    print(classification_report(y_Test, y_pred_sklearn, target_names=y_encoder.classes_))

    # Train and evaluate the custom AdaBoost SAMME model
    print("\nCustom AdaBoost Classifier:")
    custom_adaboost = AdaBoostSAMME(n_estimators=50)
    custom_adaboost.fit(X_Train_scaled, y_Train)
    y_pred_custom = custom_adaboost.predict(X_Test_scaled)
    print(classification_report(y_Test, y_pred_custom, target_names=y_encoder.classes_))


if __name__ == "__main__":
    main()
