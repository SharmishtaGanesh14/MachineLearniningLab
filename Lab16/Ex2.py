import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from xgboost import XGBRegressor, XGBClassifier

def Load_data1():
    data = pd.read_csv("../datasets/Boston.csv")
    X = data.drop(columns=["Unnamed: 0", "medv"]).values
    y = data["medv"].values
    return X, y


def Load_data2():
    data = pd.read_csv("../datasets/Weekly.csv")
    X = data.drop(columns=["Direction"]).values
    y = data["Direction"].values
    return X, y


def Data_Preprocessing_reg(X_train, X_test, y_train, y_test):
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def Data_Preprocessing_class(X_train, X_test, y_train, y_test):
    y_encoder = LabelEncoder()
    y_train = y_encoder.fit_transform(y_train)
    y_test = y_encoder.transform(y_test)
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, y_encoder


def regression_report(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    report = f"""
    Regression Report:
    ----------------------------
    Mean Squared Error (MSE)  : {mse:.4f}
    Mean Absolute Error (MAE) : {mae:.4f}
    R² Score                  : {r2:.4f}
    """
    print(report)


def manual_kfold_cv(model, X, y, n_splits=5, is_classification=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if is_classification:
            scores.append(accuracy_score(y_val, y_pred))
        else:
            scores.append(r2_score(y_val, y_pred))
    return np.mean(scores)


def main():
    # Regression Task
    X, y = Load_data1()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=323)
    X_train_scaled, X_test_scaled, y_train, y_test = Data_Preprocessing_reg(X_train, X_test, y_train, y_test)

    # there is an automatic function for hyperparameter tuning as well - called GridSearchCV
    param_grid_reg = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 1.0},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 1.0}
    ]

    best_score = -np.inf
    best_model = None
    for params in param_grid_reg:
        model = XGBRegressor(**params, random_state=42)
        score = manual_kfold_cv(model, X_train_scaled, y_train)
        if score > best_score:
            best_score = score
            best_model = model

    best_model.fit(X_train_scaled, y_train)
    y_pred1 = best_model.predict(X_test_scaled)
    print("Best parameters for Regression:", best_model.get_params())
    regression_report(y_test, y_pred1)

    # Classification Task
    X, y = Load_data2()
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.30, random_state=323)
    X_train_scaled1, X_test_scaled1, y_train1, y_test1, y_encoder = Data_Preprocessing_class(X_train1, X_test1,
                                                                                             y_train1, y_test1)

    param_grid_class = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 1.0},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 1.0}
    ]

    best_score = -np.inf
    best_model = None
    for params in param_grid_class:
        model = XGBClassifier(**params, random_state=42)
        score = manual_kfold_cv(model, X_train_scaled1, y_train1, is_classification=True)
        if score > best_score:
            best_score = score
            best_model = model

    best_model.fit(X_train_scaled1, y_train1)
    y_pred2 = best_model.predict(X_test_scaled1)
    print("Best parameters for Classification:", best_model.get_params())
    print(classification_report(y_test1, y_pred2, output_dict=False, target_names=y_encoder.classes_))


if __name__ == "__main__":
    main()
