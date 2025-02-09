import time
import os
import pandas as pd
import numpy as np
from pandas import set_option
from pandas.core.computation.ops import isnumeric
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    y2 = data["disease_score_fluct"].values
    return X, y1, y2

def Train_Test_Divide(x, y):
    up = int(x.shape[0] * 0.70)
    return x[:up], x[up:], y[:up], y[up:]

def bias_term(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

def min_max_scaling(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

# Standardization
def standardize(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return (X - X_mean) / X_std

def H_theta_calc(x, th):
    h_t_sum = []
    for i in range(x.shape[0]):
        h = 0
        for j, k in zip(th, x[i]):
            h += (j * k)
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
        th_n.append(i - alp * j)
    return np.array(th_n)


def r_square_comp(x, y, th):
    y_m = np.mean(y, axis=0)
    h = H_theta_calc(x, th)
    num = sum((i - j) ** 2 for i, j in zip(h, y))
    denom = sum((i - y_m) ** 2 for i in y)
    return 1 - (num / denom)


def gradient_descent(X_Train, X_Test, Y_Train, Y_Test, thetas):

    # X_Train=np.array([[1,2],[2,1],[3,3]])
    # new_col = np.ones((X_Train.shape[0], 1))
    # X_Train = np.hstack((new_col, X_Train))
    # Y_Train=np.array([3,4,5])
    # thetas = np.zeros((X_Train.shape[1], 1))

    iterations = 1000
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
            print(f"Divergence detected. Stopping gradient descent at {iteration}.")
            break

        if iteration > 0 and abs(cost_funcs[iteration - 1] - cost_funcs[iteration]) < 1e-4:
            # print(f"Function converged at iteration {iteration}\n")
            break

        alpha = 0.001
        thetas = Update_Params(thetas, alpha, der_cf)

    np.array(cost_funcs)
    # print(f"Final theta values:  {thetas}")
    # print("Bias Term:", float(thetas[0]))
    # print(f"Final cost function:{cost_funcs[-1]}")

    r2 = r_square_comp(X_Test, Y_Test, thetas)
    # print(f"R^2 score: {r2}")

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(len(cost_funcs)), cost_funcs, color="blue", label="Cost per iteration")
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Cost function")
    # plt.title("Cost vs iteration")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return thetas,cost_funcs,r2

def KFold_MyImplementation(X,y1):
    X = bias_term(X)
    n=len(X)
    m=len(X[0])
    folds=10
    if n%10 >= 5:
        width=int(np.ceil(n/folds))
    else:
        width=n//folds
    indices=[0]
    for i in range(10):
        if indices[i]+width<(n-1):
            indices.append(indices[i]+width)
    indices.append(n)
    Folds_Array_X = []
    Folds_Array_Y = []
    for i in range(10):
        Folds_Array_X.append(X[indices[i]:indices[i + 1], :])
        Folds_Array_Y.append(y1[indices[i]:indices[i + 1]])
    # Folds_Array_X = np.array_split(X, folds)
    # Folds_Array_Y = np.array_split(y1, folds)
    R=[]
    STAT={}
    for i in range(folds):
        # print(f"-------- FOLD {i + 1} --------")
        X_Test = Folds_Array_X[i]
        Y_Test = Folds_Array_Y[i]
        X_Train = np.vstack([fold for idx, fold in enumerate(Folds_Array_X) if idx != i])
        Y_Train = np.hstack([fold for idx, fold in enumerate(Folds_Array_Y) if idx != i])
        t,c,r=gradient_descent(X_Train, X_Test, Y_Train, Y_Test, np.zeros(X_Train.shape[1]))
        #statistical parameters to check if splitting is proper
        STAT[f"Fold{i+1}"]={"Test Mean":np.mean(X_Test,axis=0)[1],"Test n":len(X_Test),"Training Mean":np.mean(X_Train,axis=0)[1],"Training n":len(X_Train),"R^2 score": r}
        R.append(round(r, 3))
        # print("")
    print("Statistical parameters:")
    stat = pd.DataFrame(STAT).T
    stat['R^2 score'] = stat['R^2 score'].apply(lambda x: '{:.10f}'.format(x))
    print(stat)
    print(f"Average R^2: {sum(R)/len(R)}")
    #PLOTTING R2 SCORES VS FOLDS
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(R) + 1), R, marker="o", label="R^2 per fold")
    plt.xlabel("Fold")
    plt.ylabel("R^2 Score")
    plt.title("R^2 Score Across Folds")
    plt.legend()
    plt.grid(True)
    plt.show()

def KFold_SciKitLearn(X,y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores = []  # Store R-squared values
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)  # Calculate R-squared
        r2_scores.append(r2)
    # Print R-squared values for each fold
    for i, score in enumerate(r2_scores):
        print(f"Fold {i + 1}: R^2 = {score}")
    # Print the average R-squared
    print(f"Average R^2: {sum(r2_scores) / len(r2_scores):.4f}")
    #PLOTTING R2 SCORES VS FOLDS
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(r2_scores) + 1), list(map(lambda x: round(x, 3), r2_scores)), marker="o", label="R^2 per fold")
    plt.xlabel("Fold")
    plt.ylabel("R^2 Score")
    plt.title("R^2 Score Across Folds")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    X,y,y1=load_data()
    print("Before Normalisation and Standardisation")
    print(f"Max: {np.max(X,axis=0)}")
    print(f"Min: {np.min(X,axis=0)}")
    print(f"Mean: {np.mean(X,axis=0)}")
    print(f"Standard deviation: {np.std(X,axis=0)}")
    # Normalize and standardize the data
    X_normalized = min_max_scaling(X)
    print("After Normalisation")
    print("Min values:\n", np.min(X_normalized, axis=0))
    print("Max values:\n", np.max(X_normalized, axis=0))
    X_standardized = standardize(X)
    print("After standardisation")
    # ALWAYS CHECK IF MEAN IS 0 AND STD IS 1
    print("Mean values:\n", np.mean(X_standardized, axis=0))
    print(f"Standard deviation: {np.std(X_standardized, axis=0)}")

    # K-Fold Cross-Validation from Scratch
    KFold_MyImplementation(X_standardized, y)

    # K-Fold Cross-Validation using Scikit-Learn
    KFold_SciKitLearn(X_standardized, y)




