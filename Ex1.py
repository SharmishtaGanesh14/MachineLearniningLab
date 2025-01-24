import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

indices=[]
def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    y2 = data["disease_score_fluct"].values
    return X, y1, y2

def H_theta_calc(x,th):
    h_t_sum = []
    for i in range(x.shape[0]):
        h=0
        for j,k in zip(th, x[i]):
            h+=(j[0]*k)
        h_t_sum.append(h)
    return np.array(h_t_sum)

def cost_function(h,y):
    c_f=[]
    for x,y1 in zip(h,y):
        c_f.append((x-y1)**2)
    return sum(c_f)/2

def Derivative_CostF(x,y,h):
    x_t=[list(no) for no in (zip(*x))]
    sum1=[]
    for i in x_t:
        sum2=0
        for j,k,l in zip(h,y,i):
            sum2+=(j-k)*l
        sum1.append(sum2)
    return np.array(sum1)

def Update_Params(th,alp,dervs):
    th_n=[]
    for i,j in zip(th,dervs):
        th_n.append(i-alp*j)
    return np.array(th_n)


def Gradient_Descent(X_n, y, thetas):
    iterations = 100
    Cost_Array = []
    np.random.default_rng(10)
    for i in range(iterations):
        # while True:
        #     row_index = randint(0, int(X_n.shape[0] * 0.70))
        #     if row_index not in indices:
        #         indices.append(row_index)
        #         break
        row_index = np.random.randint(0, X_n.shape[0])
        X_sample = X_n[row_index].reshape(1, -1)  # Ensure 2D
        y_sample = y[row_index].reshape(1, -1)

        # Hypothesis function
        H_t = H_theta_calc(X_sample, thetas)

        # Cost function
        costf = cost_function(H_t, y_sample)
        Cost_Array.append(costf)

        # Compute the gradient
        grad_f = Derivative_CostF(X_sample, thetas, y_sample)

        alpha=0.00001
        # Update thetas
        thetas = Update_Params(thetas, alpha, grad_f)

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

    # Hypothesis function predictions on test set
    h_t = H_theta_calc(X_test, th)

    # Compute R^2 score
    SS_res = np.sum((h_t - Y_test.reshape(-1, 1)) ** 2)  # Residual Sum of Squares
    SS_tot = np.sum((Y_test.reshape(-1, 1) - np.mean(Y_test)) ** 2)  # Total Sum of Squares

    R2 = 1 - (SS_res / SS_tot)
    print(f"R^2 Score: {R2}")

if __name__ == "__main__":
    X,y,y1 = load_data()
    thetas = [[0] for i in range(X.shape[1]+1)]
    X_Train, X_test, Y_train, Y_test = Train_Test_Divide(X, y)
    main(X_Train, X_test, Y_train, Y_test, thetas)

