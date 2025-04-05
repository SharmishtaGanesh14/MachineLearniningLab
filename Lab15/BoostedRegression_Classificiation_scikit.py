import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


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


# def regression_report(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     report = f"""
#     Regression Report:
#     ----------------------------
#     Mean Squared Error (MSE)  : {mse:.4f}
#     Mean Absolute Error (MAE) : {mae:.4f}
#     R² Score                  : {r2:.4f}
#     """
#     print(report)


def nested_kfold_cv(model_class, X, y, param_grid, n_splits=5, is_classification=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        if is_classification:
            X_train,X_val,y_train,y_val,y_enc=Data_Preprocessing_class(X_train,X_val,y_train,y_val)
        else:
            X_train, X_val, y_train, y_val = Data_Preprocessing_reg(X_train, X_val, y_train, y_val)

        X_train_inner, X_test_inner, y_train_inner, y_test_inner = train_test_split(
            X_train, y_train, test_size=0.20, random_state=42)
        best_score = -np.inf
        best_model = None

        for params in param_grid:
            model = model_class(**params, random_state=42)
            model.fit(X_train_inner, y_train_inner)
            y_pred_inner = model.predict(X_test_inner)

            score = (accuracy_score(y_test_inner, y_pred_inner) if is_classification
                     else r2_score(y_test_inner, y_pred_inner))

            if score > best_score:
                best_score = score
                best_model = model

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred) if is_classification else r2_score(y_val, y_pred))
    return scores, np.mean(scores), np.std(scores)


def main():
    # Regression Task
    X, y = Load_data1()
    param_grid_reg = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 1.0},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 1.0}
    ]
    scores, mean_score, std_score = nested_kfold_cv(GradientBoostingRegressor, X, y, param_grid_reg)
    print("Regression R² scores:", scores)
    print(f"Mean R² Score: {mean_score:.4f} | Std Dev: {std_score:.4f}")

    # Classification Task
    X, y = Load_data2()
    param_grid_class = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 1.0},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 1.0}
    ]
    scores, mean_score, std_score = nested_kfold_cv(GradientBoostingClassifier, X, y, param_grid_class,
                                                    is_classification=True)
    print("Classification Accuracy scores:", scores)
    print(f"Mean Accuracy: {mean_score:.4f} | Std Dev: {std_score:.4f}")


if __name__ == "__main__":
    main()
