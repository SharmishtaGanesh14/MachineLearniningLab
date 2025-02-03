import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import time
import os
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

indices=[]

def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    y2 = data["disease_score_fluct"].values
    return X, y1, y2

def load_data2():
    X, y = fetch_california_housing(return_X_y=True)
    return X, y

def H_theta_calc(x,th):
    h_t_sum = []
    for i in range(x.shape[0]):
        h=0
        for j,k in zip(th, x[i]):
            h+=(j[0]*k)
        h_t_sum.append(h)
    return np.array(h_t_sum)
    # return x.dot(th)

def cost_function(h,y):
    c_f=[]
    for x,y1 in zip(h,y):
        c_f.append((x-y1)**2)
    return sum(c_f)/2
    # return np.sum((h - y) ** 2) / 2

def Derivative_CostF(x,y,h):
    x_t=[list(no) for no in (zip(*x))]
    sum1=[]
    for i in x_t:
        sum2=0
        for j,k,l in zip(h,y,i):
            sum2+=(j-k)*l
        sum1.append(sum2)
    return np.array(sum1)
    # # Compute the gradient of the cost function
    # errors = h - y  # Element-wise error
    # gradient = np.dot(x.T, errors)
    # return gradient

def Update_Params(th,alp,dervs):
    th_n=[]
    for i,j in zip(th,dervs):
        th_n.append(i-alp*j)
    return np.array(th_n)
    # return th - (alp * dervs)

def r_square_comp(x, y, th):
    y_m = np.mean(y, axis=0)
    h = H_theta_calc(x, th)
    num = sum((i - j) ** 2 for i, j in zip(h, y))
    denom = sum((i - y_m) ** 2 for i in y)
    return 1 - (num / denom)

def Gradient_Descent(X_n, y, thetas):
    iterations = 100000
    Cost_Array = []
    np.random.default_rng(10)
    alpha = 0.0001  # Learning rate

    for i in range(iterations):
        row_index = np.random.randint(0, X_n.shape[0])
        X_sample = X_n[row_index].reshape(1, -1)  # Ensure 2D
        y_sample = y[row_index].reshape(1, -1)

        # Hypothesis function
        H_t = H_theta_calc(X_sample, thetas)

        # Cost function for the single sample
        costf = cost_function(H_t, y_sample)
        Cost_Array.append(costf)

        # Compute the gradient
        grad_f = Derivative_CostF(X_sample, y_sample, H_t)


        # # Stopping criteria
        # if i > 0 and abs(np.mean(Cost_Array[-3:]) - Cost_Array[i]) < 10e-4:
        #     print(f"The cost function is maximised at iterations {i}")
        #     break

        # Update thetas
        thetas = Update_Params(thetas, alpha, grad_f)

    return thetas, Cost_Array

def Train_Test_Divide(x, y):
    up = int(x.shape[0] * 0.70)
    return x[:up], x[up:], y[:up], y[up:]


def main(X_train, X_test, Y_train, Y_test,thetas):

    # Normalize training data
    X_Train_mean = np.mean(X_train, axis=0)
    X_Train_std = np.std(X_train, axis=0)
    X_Train = (X_train - X_Train_mean) / X_Train_std
    new_col = np.ones((X_Train.shape[0], 1))
    X_Train = np.hstack((new_col, X_Train))

    # Normalize test data using training mean and std
    X_Test = (X_test - X_Train_mean) / X_Train_std
    new_col2 = np.ones((X_Test.shape[0], 1))
    X_Test = np.hstack((new_col2, X_Test))

    # Y_train_mean = np.mean(Y_train, axis=0)
    # Y_train_std = np.std(Y_train, axis=0)
    # Y_Train = (Y_train - Y_train_mean) / Y_train_std
    # Y_Test=(Y_test - Y_train_mean) / Y_train_std

    th, arr = Gradient_Descent(X_Train, Y_train.reshape(-1, 1), thetas)
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

    r2=r_square_comp(X_Test,Y_test,th)
    # Compute R^2 score
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    # # Simulated data
    # # y-diseas
    # # y-disease_fluct
    # X,y,y1 = load_data()

    # California Dataset
    X, y = load_data2()
    thetas = np.zeros((X.shape[1] + 1, 1))
    X_Train, X_test, Y_train, Y_test = Train_Test_Divide(X, y)
    main(X_Train, X_test, Y_train, Y_test, thetas)

