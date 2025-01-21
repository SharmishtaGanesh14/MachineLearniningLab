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
from sklearn.metrics import mean_squared_error

def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    y2 = data["disease_score_fluct"].values
    return X, y1, y2

def load_data2():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    X = data["age"].values
    y = data["disease_score_fluct"].values
    return X, y

# def Train_Test_Divide(x, y):
#     up = int(x.shape[0] * 0.70)
#     return x[:up], x[up:], y[:up], y[up:]

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

def scaling(X_Train, X_Test):
    scaler = StandardScaler()  # Initialize the scaler
    scaler.fit(X_Train)  # Fit the scaler on training data (learn mean and variance)
    X_train_scaled = scaler.transform(X_Train)  # Apply scaling to training features
    X_test_scaled = scaler.transform(X_Test)  # Apply scaling to testing features
    # X_mean = np.mean(X_Train, axis=0)
    # X_std = np.std(X_Train, axis=0)
    # X_Train = (X_Train - X_mean) / X_std
    new_col = np.ones((X_Train.shape[0], 1))
    X_Train = np.hstack((new_col, X_train_scaled))

    # X_mean2 = np.mean(X_Test, axis=0)
    # X_std2 = np.std(X_Test, axis=0)
    # X_Test = (X_Test - X_mean2) / X_std2
    new_col2 = np.ones((X_Test.shape[0], 1))
    X_Test = np.hstack((new_col2, X_test_scaled))
    return X_Train, X_Test

def gradient_descent(X_Train, X_Test, Y_Train, Y_Test, thetas):

    # X_Train=np.array([[1,2],[2,1],[3,3]])
    # new_col = np.ones((X_Train.shape[0], 1))
    # X_Train = np.hstack((new_col, X_Train))
    # Y_Train=np.array([3,4,5])
    # thetas = np.zeros((X_Train.shape[1], 1))

    iterations = 100000
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

        if iteration > 0 and abs(cost_funcs[iteration - 1] - cost_funcs[iteration]) < 1e-12:
            print(f"Function converged at iteration {iteration}\n")
            break

        # updating parameters
        if iteration < 50:
            alpha = 0.01
        else:
            alpha = 0.001
        thetas = Update_Params(thetas, alpha, der_cf)

    np.array(cost_funcs)
    print(f"Final theta values:  {thetas}")
    print("Bias Term:", float(thetas[0]))
    print(f"Final cost function:{cost_funcs[-1]}")

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
    return thetas


def Normal_Equation(X, y):
    thetas = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    return np.array(thetas)


def plotting(x_feature,pred1,pred2,pred3,y):

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each prediction on the same graph
    plt.scatter(x_feature,y,label="Actual Data")
    plt.plot(x_feature, pred1, label="Gradient Descent", linestyle='-', color='blue', marker='o')
    plt.plot(x_feature, pred2, label="Normal Equation", linestyle='--', color='green')
    plt.plot(x_feature, pred3, label="Scikit", color='orange')

    # Add labels, title, legend, and grid
    plt.xlabel('Feature age')
    plt.ylabel('Predicted Values')
    plt.title('Comparison of Linear Regression Predictions')
    plt.legend(loc='best')  # Automatically chooses the best position
    plt.grid(alpha=0.5)

    # Show the plot
    plt.show()

def main():
    # GRADIENT DESCENT
    print("\nGradient Descent\n")
    X, y1, y2 = load_data()
    thetas = [0 for i in range(X.shape[1] + 1)]

    # FOR DISEASE COLUMN WITHOUT FLUCTUATION
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        X, y1, test_size=0.30, random_state=999
    )
    X_Train,X_Test=scaling(X_Train,X_Test)
    gradient_descent(X_Train, X_Test, Y_Train, Y_Test, thetas)

    print("\nDone!\n")

    # FOR DISEASE COLUMN WITH FLUCTUATION
    X_Train2, X_Test2, Y_Train2, Y_Test2 = train_test_split(
        X, y2, test_size=0.30, random_state=999
    )
    X_Train2, X_Test2 = scaling(X_Train2, X_Test2)
    gradient_descent(X_Train2, X_Test2, Y_Train2, Y_Test2, thetas)

    print("\nDone!\n")

    # NORMAL EQUATION
    print("\nNormal Equation\n")

    # FOR DISEASE COLUMN WITHOUT FLUCTUATION
    th1 = Normal_Equation(X_Train, Y_Train)
    print("Thetas for disease without fluctuation: \n", th1)
    ht1 = H_theta_calc(X_Train, th1)
    print(f"Cost function: \n{cost_function(ht1, Y_Train)}")
    print(f'R2_Value: \n{r_square_comp(X_Test, Y_Test, th1)}')

    print("\nDone!\n")

    # FOR DISEASE COLUMN WITH FLUCTUATION
    th2 = Normal_Equation(X_Train2, Y_Train2)

    print("Thetas for disease with fluctuation: \n", th2)
    ht2 = H_theta_calc(X_Train2, th2)
    print(f"Cost function: \n{cost_function(ht2, Y_Train2)}")
    print(f'R2_Value: \n{r_square_comp(X_Test2, Y_Test2, th2)}')

    print("\nDone!\n")

    # SCIKIT-LEARN
    print("\nSCIKIT-LEARN")
    # FOR DISEASE COLUMNS WITHOUT FLUCTUATION
    model1 = LinearRegression()  # Instantiate the Linear Regression model
    model1.fit(X_Train, Y_Train)  # Train the model using training data
    ty1_pred = model1.predict(X_Test)  # Predict disease scores for test data
    r21 = r2_score(Y_Test, ty1_pred)  # Calculate R² score
    weights = model1.coef_  # theta values (weights)
    bias = model1.intercept_  # intercept (bias)
    print("\nWeights:", weights)
    print("Bias:", bias)
    print(f"R2 score: {r21:.2f} (closer to 1 is better)")

    # FOR DISEASE COLUMNS WITH FLUCTUATION
    model2 = LinearRegression()  # Recreate the model
    model2.fit(X_Train2, Y_Train2)  # Train using the second target
    ty2_pred = model2.predict(X_Test2)  # Predict disease-score fluctuations
    r22 = r2_score(Y_Test2, ty2_pred)  # Calculate R² for second model
    weights2 = model2.coef_  # theta values (weights)
    bias2 = model2.intercept_  # intercept (bias)
    print("\nWeights:", weights2)
    print("Bias:", bias2)
    print(f"R2 score: {r22:.2f} (closer to 1 is better)")
    weights_with_bias = np.insert(weights2, 0, bias2)


    # PLOTTING FOR LINEAR REGRESSION

    X2,y2=load_data2()
    X_Train3, X_Test3, Y_Train3, Y_Test3 = train_test_split(
        X2, y2, test_size=0.30, random_state=999
    )
    X_Train3, X_Test3 = scaling(X_Train3.reshape(-1,1), X_Test3.reshape(-1,1))

    t=gradient_descent(X_Train3, X_Test3, Y_Train3, Y_Test3, thetas)

    th3 = Normal_Equation(X_Train3, Y_Train3)

    model3 = LinearRegression()  # Recreate the model
    model3.fit(X_Train3, Y_Train3)  # Train using the second target
    ty3_pred = model3.predict(X_Test3)  # Predict disease-score fluctuations
    weights3 = model3.coef_  # theta values (weights)
    bias3 = model3.intercept_
    weights3[0]=bias3

    y_pred1=H_theta_calc(X_Train3,t)
    y_pred2=H_theta_calc(X_Train3,th3)
    y_pred3 = H_theta_calc(X_Train3, weights3)

    plotting(X_Train3[:,1],y_pred1,y_pred2,y_pred3,Y_Train3)

if __name__ == "__main__":
    main()



# Weights: [ 0.         42.89072639  3.53135545 15.26979029 15.64970013  0.49089104]
# Bias: 816.7399047619048
# R2 score: 1.00 (closer to 1 is better)
#
# Weights: [  0.          42.28637395   0.75307786  27.64393671  13.46280918
#  -10.68633597]
# Bias: 812.6021237651055
# R2 score: 0.57 (closer to 1 is better)