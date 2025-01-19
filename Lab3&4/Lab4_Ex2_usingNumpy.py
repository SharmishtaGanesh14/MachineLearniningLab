import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    y2 = data["disease_score_fluct"].values
    return X, y1, y2


def H_theta(X, thetas):
    # Vectorized computation of H_theta
    H_theta_f = X.dot(thetas)
    return H_theta_f


def C_F(H, y):
    E = (H - y)
    Costf = (E.T.dot(E)) / 2   # Mean squared error
    return Costf.flatten()[0]  # Return as a scalar


def Grad(X, thetas, y):
    H = H_theta(X, thetas)
    grad_f = (X.T.dot(H - y)) / 2
    return grad_f


def update_parameter(th, alp, gf):
    theta_new = th - alp * gf
    return theta_new


def Gradient_Descent(X_new, y1, thetas, alpha, iterations):
    Cost_Array = []

    for i in range(iterations):
        # Hypothesis function
        H_t = H_theta(X_new, thetas)

        # Cost function
        cost_f = C_F(H_t, y1)
        Cost_Array.append(cost_f)

        # Compute the gradient
        grad_f = Grad(X_new, thetas, y1)

        # Update thetas
        thetas = update_parameter(thetas, alpha, grad_f)

        if i > 0 and abs(Cost_Array[i - 1] - Cost_Array[i - 2]) < 1e-6:
            break

    return thetas, Cost_Array


def Train_Test_Divide(x, y):
    up = int(x.shape[0] * 0.70)
    return x[:up], x[up:], y[:up], y[up:]


def main(X_Train, X_test, Y_train, Y_test,thetas):


    # Normalize training data
    X_Train_mean = np.mean(X_Train, axis=0)
    X_Train_std = np.std(X_Train, axis=0)
    X_Train = (X_Train - X_Train_mean) / X_Train_std
    new_col = np.ones((X_Train.shape[0], 1))
    X_Train = np.hstack((new_col, X_Train))

    # Normalize test data using training mean and std
    X_test = (X_test - X_Train_mean) / X_Train_std
    new_col2 = np.ones((X_test.shape[0], 1))
    X_test = np.hstack((new_col2, X_test))

    n = X_Train.shape[0]
    alpha = 0.01
    iterations = 1000
    th, arr = Gradient_Descent(X_Train, Y_train.reshape(-1, 1), thetas, alpha, iterations)
    print(f"Thetas: {th.flatten()}")
    print(f"Value of minimized cost function: {arr[-1]}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(arr)), arr, color="blue", label="Cost per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function Value")
    plt.title("Cost Function vs. Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Hypothesis function predictions on test set
    h_t = H_theta(X_test, th)

    # Compute R^2 score
    SS_res = np.sum((h_t - Y_test.reshape(-1, 1)) ** 2)  # Residual Sum of Squares
    SS_tot = np.sum((Y_test.reshape(-1, 1) - np.mean(Y_test)) ** 2)  # Total Sum of Squares

    R2 = 1 - (SS_res / SS_tot)
    print(f"R^2 Score: {R2}")


if __name__ == "__main__":
    X, y1, y2 = load_data()
    thetas = np.zeros((X.shape[1] + 1, 1))

    # Train-test divide - disease
    X_Train, X_test, Y_train, Y_test = Train_Test_Divide(X, y1)
    main(X_Train, X_test, Y_train, Y_test,thetas)

    # Train-test divide - disease fluct
    X_Train, X_test, Y_train, Y_test = Train_Test_Divide(X, y2)
    main(X_Train, X_test, Y_train, Y_test, thetas)