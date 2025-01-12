import time
import os
import pandas as pd
import numpy as np
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
    """
    Load the California Housing dataset.
    Returns:
        X (numpy.ndarray): Feature matrix (independent variables).
        y (numpy.ndarray): Target vector (dependent variable).
    """
    X, y = fetch_california_housing(return_X_y=True)
    return X, y

def main():
    """
    Main function that performs the following tasks:
    1. Load the dataset.
    2. Split it into training and test sets.
    3. Scale the features for standardization.
    4. Train a Linear Regression model on the training set.
    5. Evaluate the model on the test set using the R2 score.
    """
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

# Entry point of the script
if __name__ == "__main__":
    main()