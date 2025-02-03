from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def scaling(X_train, X_test):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_Train = (X_train - X_mean) / X_std
    new_col = np.ones((X_Train.shape[0], 1))
    X_Train = np.hstack((new_col, X_Train))
    X_Test = (X_test - X_mean) / X_std
    new_col2 = np.ones((X_Test.shape[0], 1))
    X_Test = np.hstack((new_col2, X_Test))
    return X_Train,X_Test

def Train_Test_Divide(x, y):
    up = int(x.shape[0] * 0.70)
    return x[:up], x[up:], y[:up], y[up:]

def load():
    data=pd.read_csv("data.csv")
    data.fillna(0,inplace=True)
    x=data.drop(columns=["diagnosis","id","Unnamed: 32"]).values
    data["diagnosis_binary"]=data["diagnosis"].map({'M':1,'B':0})
    y=data["diagnosis_binary"].values
    return x,y

def thetaX(x,th):
    thX=x.dot(th)
    return thX

def log_likelihood_fn(g,y):
    L_th=0
    for g1,y1 in zip(g,y):
        L_th+=y1*np.log(g1)+(1-y1)*np.log(1-g1)
    return L_th

def partial_derivative(x,y,yHat):
    # xT=x.T
    # dervs=[]
    # for row in xT:
    #     sumofG=0
    #     for col,yi,yHi in zip(row,y,yHat):
    #         sumofG+=((yi-yHi)*col)
    #     dervs.append(sumofG)
    dervs = x.T @ (y - yHat)
    return np.array(dervs)

def update_parameter(thetas,alpha,part_drvs):
    updated_thetas=thetas+alpha*part_drvs
    return updated_thetas

def limit_conversion(g):
    for i in range(len(g)):
        g[i]=round(g[i])
    return g

def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    my_accuracy = (correct / len(y_true)) * 100
    return my_accuracy

def LogisticsRegression():
    np.set_printoptions(threshold=sys.maxsize)

    x,y=load()

    # Train-Test split
    # X_train, X_test, y_train, y_test=Train_Test_Divide(x,y)
    thetas=np.zeros((x.shape[1])+1)
    #
    # # Scaling the data
    # x_train_scaled,x_test_scaled=scaling(X_train, X_test)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=999)

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)
    new_col = np.ones((x_train_scaled.shape[0], 1))
    x_train_scaled = np.hstack((new_col, x_train_scaled))
    new_col2 = np.ones((x_test_scaled.shape[0], 1))
    x_test_scaled = np.hstack((new_col2, x_test_scaled))

    iterations=1000
    cost_func = []

    for i in range(iterations):

        # The linear function
        t_x=thetaX(x_train_scaled,thetas)

        # Sigmoid function
        g_z=1/(1+np.exp(-t_x))
        epsilon = 1e-15  # Small value to avoid log(0)
        g_z = np.clip(g_z, epsilon, 1 - epsilon)

        # cost function
        # log likelihood function
        l_th = log_likelihood_fn(g_z, y_train)
        cost_func.append(l_th)

        if i>0 and abs(np.mean(cost_func[-5:])-cost_func[i])<10e-3:
            print(f"The cost function is maximised at iterations {i}")
            break

        # partial derivative
        part_drvs=partial_derivative(x_train_scaled,y_train,g_z)


        alpha = 0.001

        thetas = update_parameter(thetas,alpha,part_drvs)

    t_x_test = thetaX(x_test_scaled, thetas)
    g_z_test = 1 / (1 + np.exp(-t_x_test))
    y_predicted_test=limit_conversion(g_z_test)
    acc=accuracy(y_test,y_predicted_test)

    # plotting
    sortG_z={val1:val2 for val1,val2 in zip(t_x,g_z)}
    t_x2=sorted(sortG_z)
    g_z2=[]
    for v in t_x2:
        g_z2.append(sortG_z[v])

    print(f'Final Accuracy: {acc}')
    print(f"Final cost function: {cost_func[-1]}")
    print(f"Final thetas: {thetas[1:]}")
    print(f"Final Bias: {thetas[0]}")

    # Derivative of sigmoid function
    gdashz= [g*(1-g) for g in g_z2]

    plt.figure(figsize=(8, 5))
    plt.plot(np.array(t_x2), np.array(g_z2), label="Sigmoid Curve")
    plt.plot(np.array(t_x2), np.array(gdashz), label="Partial derivative Curve")
    plt.xlabel("z (Linear Combination)")
    plt.ylabel("Sigmoid Output g(z)/g'(z)")
    plt.title("Logistic regression")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ACCURACY, PRECISION AND F1 SCORE


def ScikitlearnLogisticReg():

    # Load data
    X, y = load()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Print classification metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

    # Print model parameters
    print(f"Theta values (coefficients): {model.coef_}")
    print(f"Bias term (intercept): {model.intercept_}")


if __name__=="__main__":
    ScikitlearnLogisticReg()
    print("--------------------------------------------")
    LogisticsRegression()



