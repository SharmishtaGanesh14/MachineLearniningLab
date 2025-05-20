# import pandas as pd
# from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score, \
#     confusion_matrix
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import numpy as np
# from xgboost import XGBRegressor, XGBClassifier


# def Load_data1():
#     data = pd.read_csv("../datasets/Boston.csv")
#     X = data.drop(columns=["Unnamed: 0", "medv"]).values
#     y = data["medv"].values
#     return X, y


# def Load_data2():
#     data = pd.read_csv("../datasets/Weekly.csv")
#     X = data.drop(columns=["Direction"]).values
#     y = data["Direction"].values
#     return X, y


# def Data_Preprocessing_reg(X_train, X_test, y_train, y_test):
#     scaling = StandardScaler()
#     X_train_scaled = scaling.fit_transform(X_train)
#     X_test_scaled = scaling.transform(X_test)
#     return X_train_scaled, X_test_scaled, y_train, y_test


# def Data_Preprocessing_class(X_train, X_test, y_train, y_test):
#     y_encoder = LabelEncoder()
#     y_train = y_encoder.fit_transform(y_train)
#     y_test = y_encoder.transform(y_test)
#     scaling = StandardScaler()
#     X_train_scaled = scaling.fit_transform(X_train)
#     X_test_scaled = scaling.transform(X_test)
#     return X_train_scaled, X_test_scaled, y_train, y_test, y_encoder


# def manual_kfold_cv(model, X, y, param_grid, n_splits=5, is_classification=False):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     scores = []

#     for train_index, val_index in kf.split(X):
#         X_train, X_val = X[train_index], X[val_index]
#         y_train, y_val = y[train_index], y[val_index]
#         if is_classification:
#             X_train, X_val, y_train, y_val, y_enc = Data_Preprocessing_class(X_train, X_val, y_train, y_val)
#         else:
#             X_train, X_val, y_train, y_val = Data_Preprocessing_reg(X_train, X_val, y_train, y_val)

#         # Split the training data into train and validation for hyperparameter tuning
#         X_train_split, X_tune, y_train_split, y_tune = train_test_split(X_train, y_train, test_size=0.2,
#                                                                         random_state=42)

#         best_score = -np.inf
#         best_model = None
#         # Perform hyperparameter tuning on the train_split and validation (tune) data
#         for params in param_grid:
#             model.set_params(**params)
#             model.fit(X_train_split, y_train_split)
#             y_pred_tune = model.predict(X_tune)

#             if is_classification:
#                 score = accuracy_score(y_tune, y_pred_tune)
#             else:
#                 score = r2_score(y_tune, y_pred_tune)

#             if score > best_score:
#                 best_score = score
#                 best_model = model

#         # Train the model with the best hyperparameters on the entire training data
#         best_model.fit(X_train, y_train)
#         y_pred = best_model.predict(X_val)

#         # Evaluate model on validation set
#         if is_classification:
#             scores.append(accuracy_score(y_val, y_pred))
#         else:
#             scores.append(r2_score(y_val, y_pred))

#     # Return the mean score and standard deviation of the scores
#     return np.mean(scores), np.std(scores)


# def main():
#     # Regression Task
#     X, y = Load_data1()

#     param_grid_reg = [
#         {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
#         {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 1.0},
#         {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8},
#         {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 1.0}
#     ]

#     # Use the manual k-fold cross-validation to train and tune hyperparameters
#     mean_score, std_score = manual_kfold_cv(XGBRegressor(random_state=42), X, y, param_grid_reg,
#                                             is_classification=False)
#     print(f"Mean R² Score for Regression: {mean_score:.4f}")
#     print(f"Standard Deviation of R² Score: {std_score:.4f}")

#     # Classification Task
#     Xt, yt = Load_data2()

#     param_grid_class = [
#         {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
#         {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 1.0},
#         {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8},
#         {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 1.0}
#     ]

#     # Use the manual k-fold cross-validation to train and tune hyperparameters for classification
#     mean_score_class, std_score_class = manual_kfold_cv(XGBClassifier(random_state=42), Xt, yt, param_grid_class,
#                                                         is_classification=True)
#     print(f"Mean Accuracy for Classification: {mean_score_class:.4f}")
#     print(f"Standard Deviation of Accuracy: {std_score_class:.4f}")

# if __name__ == "__main__":
#     main()



import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
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


def main():
    ### REGRESSION ###
    X, y = Load_data1()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, y_train, y_test = Data_Preprocessing_reg(X_train, X_test, y_train, y_test)

    param_grid_reg = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    reg_model = GridSearchCV(XGBRegressor(random_state=42), param_grid_reg, cv=5, scoring='r2')
    reg_model.fit(X_train_scaled, y_train)
    best_reg = reg_model.best_estimator_
    y_pred = best_reg.predict(X_test_scaled)

    print("=== Regression Results ===")
    print("Best Parameters:", reg_model.best_params_)
    print("R2 Score:", r2_score(y_test, y_pred))

    ### CLASSIFICATION ###
    Xt, yt = Load_data2()
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, test_size=0.2, random_state=42)
    Xt_train_scaled, Xt_test_scaled, yt_train, yt_test, enc = Data_Preprocessing_class(Xt_train, Xt_test, yt_train, yt_test)

    param_grid_class = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    clf_model = GridSearchCV(XGBClassifier(random_state=42), param_grid_class, cv=5, scoring='accuracy')
    clf_model.fit(Xt_train_scaled, yt_train)
    best_clf = clf_model.best_estimator_
    y_pred_class = best_clf.predict(Xt_test_scaled)

    print("\n=== Classification Results ===")
    print("Best Parameters:", clf_model.best_params_)
    print("Accuracy Score:", accuracy_score(yt_test, y_pred_class))

if __name__ == "__main__":
    main()

