import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from statistics import mean


def load_data():
    data = pd.read_csv("breast_cancer.csv")
    data.fillna(0, inplace=True)
    X = data.drop(columns=["diagnosis", "id", "Unnamed: 32"]).values
    data["diagnosis_binary"] = data["diagnosis"].map({'M': 1, 'B': 0})
    y = data["diagnosis_binary"].values
    return X, y
    # data = pd.read_csv("../Lab7/Sonar.csv", header=None)
    # column_names = [f"Frequency{i}" for i in range(1, data.shape[1])] + ["Class"]
    # data.columns = column_names
    # y = data["Class"].map({'M': 1, 'R': 0})  # Map target variable to binary
    # X = data.drop("Class", axis=1)
    # return X, y


# Custom L1 and L2 norm implementations
def L2_norm(thetas):
    lam = 0.1
    return lam * np.sum(thetas ** 2)


def L1_norm(thetas):
    lam = 0.1
    return lam * np.sum(np.abs(thetas))

def without_regularisation():
    X, y = load_data()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # lambda_values = np.logspace(-7, -3, 10)
    accuracies = []

    model = LogisticRegression(max_iter=5000,penalty=None)  # Increased max_iter
    model.fit(X_train_scaled, y_train)
    model_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, model_pred)

    # Store results
    accuracies.append(acc)

    return accuracies,model.coef_


def with_regularisation(lambda_values):
    X, y = load_data()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge_accuracies = []
    lasso_accuracies = []

    # for lam in lambda_values:
    C_value = 1/2.8   # C is the inverse of lambda in scikit-learn

    # Ridge (L2)
    ridge_model = LogisticRegression(penalty='l2', solver='liblinear', C=C_value,
                                     max_iter=5000)  # Increased max_iter
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_test_scaled)
    ridge_acc = accuracy_score(y_test, ridge_pred)

    # Lasso (L1)
    lasso_model = LogisticRegression(penalty='l1', solver='saga', C=C_value,
                                     max_iter=5000)  # Changed solver to 'saga'
    lasso_model.fit(X_train_scaled, y_train)
    lasso_pred = lasso_model.predict(X_test_scaled)
    lasso_acc = accuracy_score(y_test, lasso_pred)

    # Store results
    ridge_accuracies.append(ridge_acc)
    lasso_accuracies.append(lasso_acc)

    return ridge_accuracies,lasso_accuracies,ridge_model.coef_,lasso_model.coef_

def main():
    lambda_values = np.logspace(-1, 1, 10)
    acc, coeff = without_regularisation()
    r_a,l_a,r_coeff,l_coeff=with_regularisation(lambda_values)

    print(f"WITH REGULARISATION")
    print(f"Ridge accuracies: {r_a}")
    print(f"Ridge coefficients: {r_coeff}")
    print(f"Lasso accuracies: {l_a}")
    print(f"Lasso coefficients: {l_coeff}")
    print(f"\nWITHOUT REGULARISATION")
    print(f"Accuracies: {acc}")
    print(f"Coefficients: {coeff}")

    #L2 is better- ridge is better
    # for sonar dataset optimum lambda 1
    # for breast cancer dataset it is 2.8
    # plt.figure(figsize=(8, 6))
    # plt.plot(lambda_values, acc*len(lambda_values),label="w/o regularisation")
    # plt.plot(lambda_values, r_a, label="Ridge (L2)")
    # plt.plot(lambda_values, l_a, label="Lasso (L1)")
    # plt.xscale("log")
    # plt.xlabel("Lambda")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.title("Regularization vs Accuracy")
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
