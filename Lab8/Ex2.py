import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import numpy as np


def ordinal_encoder_scikit(X):
    """
    Encodes categorical features using scikit-learn's OrdinalEncoder.

    Args:
        X (pd.DataFrame): DataFrame containing categorical features.

    Returns:
        tuple: A tuple containing the ordinal encoded features as a NumPy array
               and the fitted OrdinalEncoder.
    """
    ordinal_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_encoded = ordinal_enc.fit_transform(X)
    X_encoded = pd.DataFrame(X_encoded, columns=X.columns, index=X.index)  # Convert back to DataFrame, keep index
    X_encoded = X_encoded.fillna(-1)  # Replace NaNs with -1
    return np.array(X_encoded), ordinal_enc


def one_hot_encoding_scikit(X):
    """
    Encodes categorical features using scikit-learn's OneHotEncoder.

    Args:
        X (pd.DataFrame): DataFrame containing categorical features.

    Returns:
        tuple: A tuple containing the one-hot encoded features as a DataFrame
               and the fitted OneHotEncoder.
    """
    onehot_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = onehot_enc.fit_transform(X)
    encoded_df = pd.DataFrame(
        X_encoded, columns=onehot_enc.get_feature_names_out(X.columns), index=X.index
    )
    return encoded_df, onehot_enc


def main():
    """
    Main function to load data, split it, apply ordinal and one-hot encodings,
    train logistic regression models, and evaluate their performance.
    """
    columns = ["Age", "Menopause", "Tumor_Size", "Inv_Nodes", "Node_Caps",
               "Deg_Malig", "Breast", "Breast_Quad", "Irradiat", "Class"]

    # Load data
    data = pd.read_csv("breast_cancer_encoding.csv", header=None, names=columns)

    # Split dataset before encoding
    X = data.drop(columns=["Class"])
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.30, random_state=999)

    # Apply Ordinal Encoding
    X_train_ordinal, ordinal_enc = ordinal_encoder_scikit(X_train)
    X_test_ordinal = ordinal_enc.transform(X_test)
    X_test_ordinal = pd.DataFrame(X_test_ordinal, columns=X_test.columns, index=X_test.index)  # Keep index
    X_test_ordinal = X_test_ordinal.fillna(-1)  # Replace NaNs with -1
    X_test_ordinal = np.array(X_test_ordinal)

    # Encode target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Train Logistic Regression with Ordinal Encoding
    model1 = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
    model1.fit(X_train_ordinal, y_train_encoded)
    yhat1 = model1.predict(X_test_ordinal)
    accuracy1 = accuracy_score(y_test_encoded, yhat1)
    print('Ordinal Encoding Accuracy: %.2f' % (accuracy1 * 100))

    # Apply One-Hot Encoding
    X_train_onehot, onehot_encoder = one_hot_encoding_scikit(X_train)
    X_test_onehot = onehot_encoder.transform(X_test)  # Transform test set

    # Convert the one-hot encoded test set to a DataFrame, keeping the index
    X_test_onehot = pd.DataFrame(X_test_onehot, columns=onehot_encoder.get_feature_names_out(X_test.columns), index=X_test.index)

    # Train Logistic Regression with One-Hot Encoding
    model2 = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
    model2.fit(X_train_onehot, y_train_encoded)  # Use One-Hot Encoded data
    yhat2 = model2.predict(X_test_onehot)
    accuracy2 = accuracy_score(y_test_encoded, yhat2)
    print('One-Hot Encoding Accuracy: %.2f' % (accuracy2 * 100))


if __name__ == "__main__":
    main()

