# Import necessary libraries
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt


def main():
    # Fetch the California housing dataset as a pandas DataFrame
    california_housing = fetch_california_housing(as_frame=True)

    # Optional: Uncomment the following lines for data exploration
    # View dataset description
    print(california_housing.DESCR)

    # View first few rows of the dataset
    california_housing.frame.head()

    # View data (independent variables) and target (dependent variable) separately
    california_housing.data.head()
    california_housing.target.head()

    # Get dataset info (columns, non-null count, types, memory usage)
    california_housing.frame.info()

    # Visualize dataset distribution using histograms
    california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()

    # Analyze descriptive statistics for specific features
    features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
    print(california_housing.frame[features_of_interest].describe())

    # Import seaborn for visualization
    import seaborn as sns

    #Plot median house values on a spatial map
    sns.scatterplot(
        data=california_housing.frame,
        x="Longitude",
        y="Latitude",
        size="MedHouseVal",
        hue="MedHouseVal",
        palette="viridis",
        alpha=0.5,
    )
    plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95), loc="upper left")
    plt.title("Median house value depending of\n their spatial location")
    plt.show()

    # Randomly sample 500 data points to improve visualization
    import numpy as np
    rng = np.random.RandomState(0)
    indices = rng.choice(
        np.arange(california_housing.frame.shape[0]), size=500, replace=False
    )
    sns.scatterplot(
        data=california_housing.frame.iloc[indices],
        x="Longitude",
        y="Latitude",
        size="MedHouseVal",
        hue="MedHouseVal",
        palette="viridis",
        alpha=0.5,
    )
    plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 1), loc="upper left")
    _ = plt.title("Median house value depending of\n their spatial location")
    plt.show()

    # Import pandas for data manipulation
    import pandas as pd

    # Drop spatial columns for further analysis
    columns_drop = ["Longitude", "Latitude"]
    subset = california_housing.frame.iloc[indices].drop(columns=columns_drop)

    # Bin the target variable (median house values) into quantiles
    subset["MedHouseVal"] = pd.qcut(subset["MedHouseVal"], 6, retbins=False)
    subset["MedHouseVal"] = subset["MedHouseVal"].apply(lambda x: x.mid)

    # Create pairplots to visualize feature relationships
    _ = sns.pairplot(data=subset, hue="MedHouseVal", palette="viridis")

    # Import libraries for modeling
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_validate

    # Define a range of alpha values for Ridge regression
    alphas = np.logspace(-3, 1, num=30)

    # Create a pipeline for scaling and Ridge regression with cross-validation
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

    # Perform cross-validation to evaluate the model
    cv_results = cross_validate(
        model,
        california_housing.data,
        california_housing.target,
        return_estimator=True,
        n_jobs=2,
    )

    # Compute and print mean R2 score
    score = cv_results["test_score"]
    print(f"R2 score: {score.mean():.3f} ± {score.std():.3f}")

    # Extract Ridge model coefficients and plot them
    coefs = pd.DataFrame(
        [est[-1].coef_ for est in cv_results["estimator"]],
        columns=california_housing.feature_names,
    )

    # Customize plot colors and visualize coefficient variability
    color = {"whiskers": "black", "medians": "black", "caps": "black"}
    coefs.plot.box(vert=False, color=color)
    plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
    _ = plt.title("Coefficients of Ridge models\n via cross-validation")


if __name__ == "__main__":
    main()
