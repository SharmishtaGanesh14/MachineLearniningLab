import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,pi
import pandas as pd

def Ex1():
    A=np.array([[1,2,3],[4,5,6]])
    At=A.transpose()
    print(At)
    print(At.dot(A))

def Ex2():
    x=np.linspace(-100,100,100)
    y=2*x+3
    plt.figure(figsize=(8,6))
    plt.plot(x,y,label="y = 2x + 3", color='violet')
    plt.title("Plot of y = 2x + 3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=0.8, linestyle="--")
    plt.axvline(0, color='black', linewidth=0.8, linestyle="--")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()

def Ex3():
    x=np.linspace(-10,10,100)
    y=2*x**2+3*x+4
    plt.figure(figsize=(8,6))
    plt.plot(x,y,label="y = 2x^2 + 3x + 4",color="violet")
    plt.title("Plot of y = 2x^2 + 3x + 4")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

def Ex4():
    x=np.linspace(-100,100,100)
    me=0
    sd=15
    y=np.exp(-0.5*((x-me)/sd)**2)/(sd*sqrt(2*pi))
    plt.plot(x, y, color="violet")
    plt.title("Gaussian Distribution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.4)
    plt.show()

def Ex5():
    x=np.linspace(-100,100,100)
    y=x**2
    dydx=np.gradient(y,x)
    plt.figure(figsize=(8,6))
    plt.plot(x,y,label="Function f(x)=x^2",color="violet")
    plt.plot(x,dydx,label="Derivative f'(x)=2*x",color="Red")
    plt.legend()
    plt.show()

def Ex6():
    def hthetax(thetas, Xes, x0):
        h_x = thetas[0] * x0  # Bias term
        h_x += sum(t * x for t, x in zip(thetas[1:], Xes))  # Other features
        return h_x

    def get_float_input(prompt):
        """Helper function to repeatedly prompt the user until a valid float is provided."""
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Invalid input! Please enter a numeric value.")

    def get_positive_int_input(prompt):
        """Helper function to repeatedly prompt the user until a positive integer is provided."""
        while True:
            try:
                value = int(input(prompt))
                if value > 0:
                    return value
                else:
                    print("Input must be a positive integer.")
            except ValueError:
                print("Invalid input! Please enter a positive integer.")

    # Welcome message
    print("Welcome to the Model Error Calculator!")
    print("This tool calculates the squared error for a given linear hypothesis.")

    # Input: Number of samples and features
    n = get_positive_int_input("Enter the number of samples: ")
    d = get_positive_int_input("Enter the number of features: ")

    # Input: Common values
    print(f"\n--- Initial common values ---")
    x0 = get_float_input("Enter the value of x0 (common to all samples): ")

    # Input: Thetas
    thetas = []
    for i in range(d + 1):
        theta = get_float_input(f"Enter the value of theta{i}: ")
        thetas.append(theta)

    # Data table columns
    columns = [f"x{i}" for i in range(1, d + 1)] + ["Y"]
    data = []

    # Input: Feature values and target variable for each sample
    for j in range(1, n + 1):
        print(f"\n--- Sample {j} ---")
        row = []
        for i in range(1, d + 1):
            X = get_float_input(f"Enter the value for x{i}: ")
            row.append(X)
        Y = get_float_input(f"Enter the value for target variable Y: ")
        row.append(Y)
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    print("\nData Table:")
    print(df)

    # Compute squared errors
    Hes = []
    for index, row in df.iterrows():
        h = hthetax(thetas, row[:-1].values.tolist(), x0)  # Convert row to list for calculation
        y = row["Y"]
        Hes.append((h - y) ** 2)

    # Calculate total error
    E = 0.5 * sum(Hes)
    print("\nError :", E)

def main():
    # Ex1()
    # Ex2()
    # Ex3()
    # Ex4()
    # Ex5()
    Ex6()
if __name__=="__main__":
    main()