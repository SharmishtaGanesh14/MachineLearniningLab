import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"], axis=1).values
    y = data["disease_score"]
    return X, y

def load_data2():
    data = pd.read_csv("breast_cancer.csv")
    X = data.drop(columns=["diagnosis", "id", "Unnamed: 32"]).values
    data["diagnosis_binary"] = data["diagnosis"].map({'M': 1, 'B': 0})
    y = data["diagnosis_binary"].values
    return X, y

def train_regression_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

def train_classification_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc}")

def partition_dataset_based_on_BP():
    thresholds = [80, 78, 82]
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    for t in thresholds:
        upper_partition = data[data["BP"] > t]
        lower_partition = data[data["BP"] <= t]
        upper_partition.to_csv(f"data_filtered_upper{t}.csv", index=False)
        lower_partition.to_csv(f"data_filtered_lower{t}.csv", index=False)
        print(f"Partitioning done for t = {t}:")
        print(f"Lower partition (BP <= {t}): {lower_partition.shape[0]} rows")
        print(f"Upper partition (BP > {t}): {upper_partition.shape[0]} rows\n")

def main():
    # Exercise 1 - partitioning the dataset
    partition_dataset_based_on_BP()

    # Exercise 2 - Regression Decision tree
    X,y=load_data()
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=999)
    print("Mean squared error for Decision tree regressor:")
    train_regression_tree(X_Train,X_Test,y_Train,y_Test)

    # Exercise 3 - Classification Decsion Tree
    X1,y1=load_data2()
    X_Train1, X_Test1, y_Train1, y_Test1 = train_test_split(X1, y1, test_size=0.30, random_state=999)
    print("\nAccuracy value for Decision tree classification:")
    train_classification_tree(X_Train1, X_Test1, y_Train1, y_Test1)

if __name__=="__main__":
    main()