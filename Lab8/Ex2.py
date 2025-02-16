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


def custom_ordinal_encoder(X):
    """
    Custom implementation of Ordinal Encoding.
    Converts categorical values into numerical rankings.
    """
    unique_values = {}  # Dictionary to store mappings for each column

    # Iterate through each column in the DataFrame
    for col in X.columns:
        # Drop NaN values and get unique values in sorted order
        unique_categories = sorted(X[col].dropna().unique())

        # Create a mapping {category: index} for each unique value
        mapping = {val: idx for idx, val in enumerate(unique_categories)}

        # Store mapping for this column
        unique_values[col] = mapping

    # Initialize an empty DataFrame to store encoded values
    X_encoded = pd.DataFrame()

    # Iterate through each column and replace categories with their numeric rank
    for col in X.columns:
        X_encoded[col] = X[col].map(unique_values[col])  # Map categorical to numeric values
        X_encoded[col] = X_encoded[col].fillna(-1)  # Fill NaN with -1

    # Convert DataFrame to NumPy array and return along with mappings
    return np.array(X_encoded), unique_values


def custom_ordinal_transform(X, mapping):
    """
    Transform new data using the learned mapping from custom Ordinal Encoding.
    If a value is not in the mapping, it is replaced with -1.
    """
    X_encoded = pd.DataFrame()

    # Iterate through each column
    for col in X.columns:
        X_encoded[col] = X[col].map(mapping[col])  # Apply mapping
        X_encoded[col] = X_encoded[col].fillna(-1)  # Fill unknown values with -1

    return np.array(X_encoded)


def custom_one_hot_encoder(X):
    """
    Custom implementation of One-Hot Encoding.
    Converts categorical variables into binary columns.
    """
    unique_values = {}  # Dictionary to store unique values for each column
    encoded_data = []  # List to store encoded columns
    new_columns = []  # List to store new column names

    # Iterate through each column in the DataFrame
    for col in X.columns:
        # Get sorted unique values excluding NaN
        unique_categories = sorted(X[col].dropna().unique())

        # Store unique categories for this column
        unique_values[col] = unique_categories

        # Iterate through each unique category
        for val in unique_categories:
            # Create a new column name
            new_col_name = f"{col}_{val}"
            new_columns.append(new_col_name)

            # Convert column values to binary (1 if matches, else 0)
            binary_column = (X[col] == val).astype(int)
            encoded_data.append(binary_column)

    # Concatenate all binary columns into a single DataFrame
    X_encoded = pd.concat(encoded_data, axis=1)
    X_encoded.columns = new_columns  # Assign column names

    return X_encoded, unique_values


def custom_one_hot_transform(X, mapping):
    """
    Transform new data using the learned mapping from custom One-Hot Encoding.
    """
    encoded_data = []  # List to store encoded columns
    new_columns = []  # List to store new column names

    # Iterate through each column in the mapping
    for col, values in mapping.items():
        # Iterate through each known unique value
        for val in values:
            # Create a new column name
            new_col_name = f"{col}_{val}"
            new_columns.append(new_col_name)

            # Convert column values to binary (1 if matches, else 0)
            binary_column = (X[col] == val).astype(int)
            encoded_data.append(binary_column)

    # Concatenate all binary columns into a single DataFrame
    X_encoded = pd.concat(encoded_data, axis=1)
    X_encoded.columns = new_columns  # Assign column names

    return X_encoded

def main():
    columns = ["Age", "Menopause", "Tumor_Size", "Inv_Nodes", "Node_Caps",
               "Deg_Malig", "Breast", "Breast_Quad", "Irradiat", "Class"]

    data = pd.read_csv("breast_cancer_encoding.csv", header=None, names=columns)
    X = data.drop(columns=["Class"])
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.30, random_state=999)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Scikit-Learn Ordinal Encoding
    X_train_ordinal, ordinal_enc = ordinal_encoder_scikit(X_train)
    # Transform X_test
    X_test_ordinal = ordinal_enc.transform(X_test)
    # Convert to DataFrame to retain column structure
    X_test_ordinal = pd.DataFrame(X_test_ordinal, columns=X_train.columns, index=X_test.index)
    # Replace NaNs with -1 (or another meaningful value)
    X_test_ordinal = X_test_ordinal.fillna(-1).to_numpy()
    model1 = LogisticRegression(max_iter=1000)
    model1.fit(X_train_ordinal, y_train_encoded)
    accuracy1 = accuracy_score(y_test_encoded, model1.predict(X_test_ordinal))
    print('Scikit-Learn Ordinal Encoding Accuracy: %.2f' % (accuracy1 * 100))

    # Custom Ordinal Encoding
    X_train_custom_ordinal, custom_mapping = custom_ordinal_encoder(X_train)
    X_test_custom_ordinal = custom_ordinal_transform(X_test, custom_mapping)

    model2 = LogisticRegression(max_iter=1000)
    model2.fit(X_train_custom_ordinal, y_train_encoded)
    accuracy2 = accuracy_score(y_test_encoded, model2.predict(X_test_custom_ordinal))
    print('Custom Ordinal Encoding Accuracy: %.2f' % (accuracy2 * 100))

    # Scikit-Learn One-Hot Encoding
    X_train_onehot, onehot_encoder = one_hot_encoding_scikit(X_train)
    X_test_onehot = onehot_encoder.transform(X_test)
    X_test_onehot = pd.DataFrame(X_test_onehot, columns=onehot_encoder.get_feature_names_out(X_test.columns),
                                 index=X_test.index)

    model3 = LogisticRegression(max_iter=1000)
    model3.fit(X_train_onehot, y_train_encoded)
    accuracy3 = accuracy_score(y_test_encoded, model3.predict(X_test_onehot))
    print('Scikit-Learn One-Hot Encoding Accuracy: %.2f' % (accuracy3 * 100))

    # Custom One-Hot Encoding
    X_train_custom_onehot, custom_onehot_mapping = custom_one_hot_encoder(X_train)
    X_test_custom_onehot = custom_one_hot_transform(X_test, custom_onehot_mapping)

    model4 = LogisticRegression(max_iter=1000)
    model4.fit(X_train_custom_onehot, y_train_encoded)
    accuracy4 = accuracy_score(y_test_encoded, model4.predict(X_test_custom_onehot))
    print('Custom One-Hot Encoding Accuracy: %.2f' % (accuracy4 * 100))

if __name__ == "__main__":
    main()
