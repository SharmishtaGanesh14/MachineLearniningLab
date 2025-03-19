# CODE WRITTEN ON: 11/03/2025
# Implement bagging regressor and classifier using scikit-learn. Use diabetes and iris datasets.
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from Lab12.Ex12 import load_data,Decision_tree_regression_fromscratch
from Lab10and11.Ex10_11 import load_data2,decision_tree_classifier_from_scratch
from cross_val import cross_validation

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

def Data_processing_class(X_Train, X_Test, y_Train, y_Test):
    label=LabelEncoder()
    y_encoded_train=label.fit_transform(y_Train)
    y_encoded_test=label.transform(y_Test)
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_Train)
    X_test_scaled = scaling.transform(X_Test)
    return X_train_scaled, X_test_scaled, y_encoded_train, y_encoded_test

def Data_processing_reg(X_Train, X_Test, y_Train, y_Test):
    scaling=StandardScaler()
    X_train_scaled=scaling.fit_transform(X_Train)
    X_test_scaled=scaling.transform(X_Test)
    return X_train_scaled,X_test_scaled,np.array(y_Train),np.array(y_Test)

def scikit_regressor():
    # Load the dataset
    X,y=load_data()
    # from sklearn.datasets import load_diabetes
    # import pandas as pd
    # data=load_diabetes()
    # X=pd.DataFrame(data.data)
    # y=pd.Series(data.target)

    # WITHOUT CROSS VALIDATION
    # Split the data into training and testing sets
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=22)
    X_Train_t, X_Test_t, y_Train_t, y_Test_t=Data_processing_reg(X_Train, X_Test, y_Train, y_Test)
    # Create the base regressor
    regrModel = BaggingRegressor(estimator=DecisionTreeRegressor(),
                                 n_estimators=10, random_state=0)
    regs=regrModel.fit(X_Train_t,y_Train_t)
    # Make predictions on the test set
    y_pred=regrModel.predict(X_Test_t)
    # Calculate accuracy
    r2=r2_score(y_Test_t,y_pred)
    print("Whole tree R^2 avg value:")
    print("R^2 value:", r2)

    # CROSS VALIDATION ON BAGGING
    res1,res2,res3=cross_validation(BaggingRegressor(estimator=DecisionTreeRegressor(),n_estimators=10, random_state=0),X,y,cv=10,scoring="r2",classification=False)
    print(f"R2 score mean: {res1}")
    print(f"R2 standard dev: {res2}")

    # NOT NECESSARY
    # Calculate accuracy for each regressor built
    # print("Individual tree R^2 value:")
    # for i, reg in enumerate(regs):
    #     y_pred = reg.predict(X_Test)
    #     # Calculate accuracy
    #     r2 = r2_score(y_Test, y_pred)
    #     print("R^2 value " + str(i + 1), ':', r2)


def scikit_classifer():
    # WITHOUT CROSS VALIDATION
    # Load the dataset
    # X, y = load_data2()
    import pandas as pd
    data=load_iris()
    X=pd.DataFrame(data.data)
    y=pd.Series(data.target)

    # Split the data into training and testing sets
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=22)
    # X_Train, X_Test, y_Train, y_Test= Data_processing_class(X_Train, X_Test, y_Train, y_Test)

    # Create the base regressor
    classModel = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=2,min_samples_split=5,random_state=232),
                                 n_estimators=10, random_state=0)
    classifiers = classModel.fit(X_Train, y_Train)
    # Make predictions on the test set
    y_pred = classModel.predict(X_Test)
    # Calculate accuracy
    acc = accuracy_score(y_Test, y_pred)
    print("Whole tree accuracy value:")
    print("Accuracy value:", acc)
    # Calculate accuracy for each regressor built
    # print("Individual tree accuracy values:")
    # for i, classi in enumerate(classifiers):
    #     y_pred = classi.predict(X_Test)
    #     # Calculate accuracy
    #     acc = accuracy_score(y_Test, y_pred)
    #     print("Accuracy value " + str(i + 1), ':', acc)

    # CROSS VALIDATION ON BAGGING
    res1,res2,res3=cross_validation(BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=2,min_samples_split=5,random_state=232),n_estimators=10, random_state=0),X,y,cv=10,scoring="accuracy",classification=True)
    print(f"Accuracy score mean: {res1}")
    print(f"Accuracy standard dev: {res2}")

def main():
    scikit_regressor()
    scikit_classifer()

if __name__=="__main__":
    main()