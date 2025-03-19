from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from cross_val import cross_validation


def load_data_iris():
    data=load_iris()
    X=pd.DataFrame(data.data)
    y=pd.Series(data.target)
    return X,y

def load_data_diabetes():
    data=load_diabetes()
    X=pd.DataFrame(data.data)
    y=pd.Series(data.target)
    return X,y

def scikit_randomforest_regression():
    X,y=load_data_diabetes()
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3434,test_size=0.30)
    scaling=StandardScaler()
    X_train_scaled=scaling.fit_transform(X_train)
    X_test_scaled=scaling.transform(X_test)
    model=RandomForestRegressor(n_estimators=10,max_depth=2,min_samples_split=5,random_state=22)
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    print(f"Random Forest-Diabetes")
    print(f"R2 value: {r2_score(y_test,y_pred)}")

    res1, res2, res3 = cross_validation(RandomForestRegressor(n_estimators=10,max_depth=2,min_samples_split=5,random_state=22), X, y, cv=10, scoring="r2", classification=False)
    print(f"Accuracy score mean: {res1}")
    print(f"Accuracy standard dev: {res2}")

def scikit_randomforest_classification():
    X, y = load_data_iris()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3434, test_size=0.30)
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)
    model = RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=5,random_state=22)
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"Random Forest-Iris")
    print(f"Accuracy value: {accuracy_score(y_test, y_pred)}")

    res1, res2, res3 = cross_validation(
        RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=5,random_state=22), X, y, cv=10, scoring="accuracy", classification=True)
    print(f"Accuracy score mean: {res1}")
    print(f"Accuracy standard dev: {res2}")

scikit_randomforest_regression()
scikit_randomforest_classification()