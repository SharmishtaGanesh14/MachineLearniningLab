from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline


def load():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    return X, y1

def model_selection(X_train, X_val, y_train, y_val):
    training_scores = []
    validation_scores = []
    R2=[]
    degrees = [1,2,3,5]

    for deg in degrees:
        model = make_pipeline(
            PolynomialFeatures(deg),
            StandardScaler(),
            LinearRegression()
        )
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_err = mean_squared_error(y_train, y_pred_train)
        val_err = mean_squared_error(y_val, y_pred_val)
        r_square=r2_score(y_val,y_pred_val)
        R2.append(r_square)
        training_scores.append(train_err)
        validation_scores.append(val_err)

    for i,n in zip(degrees,R2):
        print(f'{i} degree polynomial: {n}')

    plt.figure(figsize=(6, 4))
    plt.plot(degrees, training_scores, marker="o", label="Training Error")
    plt.plot(degrees, validation_scores, marker="o", label="Validation Error", linestyle="dashed")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Error")
    plt.title("Train vs Validation Error")
    plt.legend()
    plt.show()


def main():
    X, y = load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    model_selection(X_train2, X_val, y_train2, y_val)


if __name__ == "__main__":
    main()
