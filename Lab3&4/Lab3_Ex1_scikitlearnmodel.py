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

# Load the dataset, extracting features (X) and targets (y1 and y2)
def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv") # Load the dataset
    X = data.drop(columns=["disease_score", "disease_score_fluct"], axis=1).values  # Features
    y1 = data["disease_score"]  # Target variable 1
    y2 = data["disease_score_fluct"]  # Target variable 2
    return X, y1, y2

def EDA():
    # Load the dataset
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")

    # Display general dataset statistics and metadata
    pd.set_option('display.max_rows', None)  # Show all rows if needed
    pd.set_option('display.max_columns', None)  # Show all columns
    print("Dataset Description:")
    print(data.describe())  # Basic statistical summary of numerical columns
    print("\nFirst 5 rows of the dataset:")
    print(data.head())  # First 5 rows for a quick preview
    print("\nData Information:")
    print(data.info())  # Overview of column types and non-null counts

    # Check for missing values
    print("\nChecking for Null Values:")
    print(data.isnull().sum())  # Count of missing values in each column

    # Show correlation between features and targets
    print("\nCorrelation Matrix:")
    print(data.corr())

    # Display specific feature values
    print("\nFeatures:")
    print(data.iloc[:, 1:-2])  # Exclude first and last two columns for features
    print("\nTarget1 and 2:")
    print(data.iloc[:, -2:])  # Last two columns as targets

    # Create histograms for each column
    print("\nDisplaying histograms for each feature:")
    data.hist(figsize=(12, 10), bins=30, edgecolor="black")  # Histogram with clear separation
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()

    # Example: Advanced data analysis with California Housing Data (for illustration)
    california_housing = fetch_california_housing(as_frame=True)  # Fetch sample data
    features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
    print("\nCalifornia Housing Feature Description:")
    print(california_housing.frame[features_of_interest].describe())  # Summary statistics

    # Scatter plot (age vs BP) with size and hue indicating disease_score
    sns.scatterplot(
        data=data,
        x="age",
        y="BP",
        size="disease_score",
        hue="disease_score",
        palette="viridis",
        alpha=0.5
    )
    plt.legend(title="Disease Score (age vs BP)", bbox_to_anchor=(1.05, 0.95), loc="upper left")
    plt.title("Disease scores based on age and BP")
    plt.show()

    # Scatter matrix to show pairwise relationships
    pd.plotting.scatter_matrix(
        data[["age", "BMI", "BP", "blood_sugar", "Gender", "disease_score", "disease_score_fluct"]],
        figsize=(10, 10), diagonal='kde')
    plt.show()

    # Subplots for visualizing relationships between age and targets
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot 1: Disease score vs. Age
    axes[0].scatter(data['age'], data['disease_score'], color='blue', alpha=0.7)
    axes[0].set_title('Age vs Disease Score')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Disease Score')

    # Plot 2: Disease score fluctuations vs. Age
    axes[1].scatter(data['age'], data['disease_score_fluct'], color='green', alpha=0.7)
    axes[1].set_title('Age vs Disease Score Fluctuations')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Disease Score Fluctuations')

    # Show both subplots
    plt.tight_layout()
    plt.show()


def main():
    # Step 1: Load the data
    X, ty1, ty2 = load_data()

    # -------- Part 1: Disease Score as Target -------- #
    # Step 2: Split the data into training and testing sets for the first target (ty1)
    X_train, X_test, ty1_train, ty1_test = train_test_split(
        X, ty1, test_size=0.30, random_state=999
    )  # 30% data as test set

    # Step 3: Feature scaling using StandardScaler
    scaler = StandardScaler()  # Initialize the scaler
    scaler.fit(X_train)  # Fit the scaler on training data (learn mean and variance)
    X_train_scaled = scaler.transform(X_train)  # Apply scaling to training features
    X_test_scaled = scaler.transform(X_test)  # Apply scaling to testing features

    # Step 4: Initialize and train the linear regression model
    model = LinearRegression()  # Instantiate the Linear Regression model
    model.fit(X_train, ty1_train)  # Train the model using training data

    # Step 5: Make predictions for test set
    ty1_pred = model.predict(X_test)  # Predict disease scores for test data

    # Step 6: Evaluate model performance using R² score
    r2 = r2_score(ty1_test, ty1_pred)  # Calculate R² score
    weights = model.coef_  # theta values (weights)
    bias = model.intercept_  # intercept (bias)
    print("\nWeights:", weights)
    print("Bias:", bias)
    cost = mean_squared_error(ty1_test, ty1_pred)
    print(f"R2 score: {r2:.2f} (closer to 1 is better)")


    # -------- Part 2: Disease-Score Fluctuations as Target -------- #
    # Step 7: Split the data for the second target (ty2)
    X_train, X_test, ty2_train, ty2_test = train_test_split(
        X, ty2, test_size=0.30, random_state=999
    )  # Recreate train-test split for second target

    # Step 8: Apply scaling (necessary even if the results do not change, for consistency)
    scaler = StandardScaler()  # Re-initialize scaler
    scaler.fit(X_train)  # Fit on new training data
    X_train_scaled = scaler.transform(X_train)  # Scale training data
    X_test_scaled = scaler.transform(X_test)  # Scale testing data

    # Step 9: Train the regression model for the second target
    model = LinearRegression()  # Recreate the model
    model.fit(X_train, ty2_train)  # Train using the second target

    # Step 10: Make predictions for the second target
    ty2_pred = model.predict(X_test)  # Predict disease-score fluctuations

    # Step 11: Evaluate performance for the second target using R² score
    r2 = r2_score(ty2_test, ty2_pred)  # Calculate R² for second model
    print("\nWeights:", weights)
    print("Bias:", bias)
    cost = mean_squared_error(ty2_test, ty2_pred)
    print(f"R2 score: {r2:.2f} (closer to 1 is better)")

    # Step 12: Insights and observations
    """
    - Scaling did not affect the R² value significantly because the linear regression model is invariant to scaling.
    - Disease_score is more predictable with the available predictors, leading to a higher R².
    - Disease_score_fluct involves more randomness or nonlinear factors, resulting in lower R².
    """


if __name__ == "__main__":
    # Perform exploratory data analysis (not detailed here)
    # EDA()
    # Run the ML pipeline
    main()
