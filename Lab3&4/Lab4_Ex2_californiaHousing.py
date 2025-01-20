import time
import os
import pandas as pd
import numpy as np
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def load_data():
    X, y = fetch_california_housing(return_X_y=True)
    return X, y

def My_Implementation():

    def Train_Test_Divide(x, y):
        up = int(x.shape[0] * 0.70)
        return x[:up], x[up:], y[:up], y[up:]

    def H_theta_calc(x, th):
        h_t_sum = []
        for i in range(x.shape[0]):
            h = 0
            for j, k in zip(th, x[i]):
                h += (j[0] * k)
            h_t_sum.append(h)
        return np.array(h_t_sum)

    def cost_function(h, y):
        c_f = []
        for x, y1 in zip(h, y):
            c_f.append((x - y1) ** 2)
        return (sum(c_f) / 2)

    def Derivative_CostF(x, y, h):
        x_t = [list(no) for no in (zip(*x))]
        sum1 = []
        for i in x_t:
            sum2 = 0
            for j, k, l in zip(h, y, i):
                sum2 += (j - k) * l
            sum1.append(sum2)
        return np.array(sum1)

    def Update_Params(th, alp, dervs):
        th_n = []
        for i, j in zip(th, dervs):
            th_n.append([i[0] - alp * j])
        return np.array(th_n)

    def r_square_comp(x, y, th):
        y_m = np.mean(y, axis=0)
        h = H_theta_calc(x, th)
        num = sum((i - j) ** 2 for i, j in zip(h, y))
        denom = sum((i - y_m) ** 2 for i in y)
        return 1 - (num / denom)

    def gradient_descent(X_Train, X_Test, Y_Train, Y_Test, thetas):
        X_mean = np.mean(X_Train, axis=0)
        X_std = np.std(X_Train, axis=0)
        X_Train = (X_Train - X_mean) / X_std

        new_col = np.ones((X_Train.shape[0], 1))
        X_Train = np.hstack((new_col, X_Train))

        X_mean2 = np.mean(X_Test, axis=0)
        X_std2 = np.std(X_Test, axis=0)
        X_Test = (X_Test - X_mean2) / X_std2
        new_col2 = np.ones((X_Test.shape[0], 1))
        X_Test = np.hstack((new_col2, X_Test))

        # X_Train=np.array([[1,2],[2,1],[3,3]])
        # new_col = np.ones((X_Train.shape[0], 1))
        # X_Train = np.hstack((new_col, X_Train))
        # Y_Train=np.array([3,4,5])
        # thetas = np.zeros((X_Train.shape[1], 1))

        iterations = 2000
        cost_funcs = []

        for iteration in range(iterations):
            # calculate hypothesis
            h_t = H_theta_calc(X_Train, thetas)

            # calculate cost function
            c_f = cost_function(h_t, Y_Train)
            cost_funcs.append(c_f)

            # calculate derivative of cost function
            der_cf = Derivative_CostF(X_Train, Y_Train, h_t)

            if iteration > 0 and (np.isnan(c_f) or cost_funcs[-1] > 1e10):
                print(f"Divergence detected. Stopping gradient descent at {iteration}.\n")
                break

            if iteration > 0 and abs(cost_funcs[iteration - 1] - cost_funcs[iteration]) < 1e-10:
                print(f"Function converged at iteration {iteration}\n")
                break

            # updating parameters
            if iteration <900:
                alpha = 0.00001
            else:
                alpha = 0.000001
            thetas = Update_Params(thetas, alpha, der_cf)

        np.array(cost_funcs)
        print(f"Final theta values:\n{thetas}\n")
        print(f"Final cost function:\n{cost_funcs[-1]}\n")

        r2 = r_square_comp(X_Test, Y_Test, thetas)
        print(f"R^2 score: {r2}")

        plt.figure(figsize=(8, 5))
        plt.plot(range(len(cost_funcs)), cost_funcs, color="blue", label="Cost per iteration")
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost function")
        plt.title("Cost vs iteration")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Load the dataset
    X, y = load_data()
    thetas = np.zeros((X.shape[1] + 1, 1))

    # Split the data into training and test sets
    X_Train, X_Test, Y_Train, Y_Test = Train_Test_Divide(X, y)
    gradient_descent(X_Train, X_Test, Y_Train, Y_Test, thetas)

def Scikitlearn():
    # Load the dataset
    X, y = load_data()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=999
    )
    # 70% of the data for training, 30% for testing; fixed randomness for reproducibility.

    ### Step 1: Scale the features ###
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training set and apply transformation to both train and test sets
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ### Step 2: Train the model ###
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model using the scaled training data
    model.fit(X_train_scaled, y_train)

    ### Step 3: Make predictions on the test set ###
    y_pred = model.predict(X_test_scaled)

    ### Step 4: Evaluate the model ###
    # Compute the R2 score (coefficient of determination)
    r2 = r2_score(y_test, y_pred)

    # Print the result
    print(f"R2 score: {r2:.2f} (closer to 1 is better)")

def main():

    My_Implementation()

    print("\n----------Done!----------\n")


    # Scikitlearn()
    #
    # print("\n----------Done!----------\n")

if __name__ == "__main__":
    main()
