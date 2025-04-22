import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def EDA(data):
    print("First 5 rows:")
    print(data.head(), "\n")

    print("Dataset Info:")
    print(data.info(), "\n")

    print("Missing Values:")
    print(data.isnull().sum(), "\n")

    print("Class Distribution:")
    print(data['v1'].value_counts(), "\n")

    # Bar plot of class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='v1', data=data)
    plt.title("Class Distribution (Ham vs Spam)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

    # Add message length as a new column
    data['length'] = data['v2'].apply(len)

    # Boxplot of message length by class
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='v1', y='length', data=data)
    plt.title("Message Length Distribution by Class")
    plt.show()

    # Descriptive stats by label
    print("Message Length Stats by Label:")
    print(data.groupby('v1')['length'].describe(), "\n")

def load_data():
    # Load the dataset
    data = pd.read_csv("spam_sms.csv", encoding='latin-1')  # Make sure to use the correct encoding if needed
    # Optional: run EDA
    EDA(data)

    # Extract features and target
    y = data["v1"]  # Target: ham or spam
    X = data["v2"]  # SMS text message
    return X, y


def data_preprocessing(X_train, X_test, y_train, y_test):
    # Encode labels: 'ham' -> 0, 'spam' -> 1
    label = LabelEncoder()
    y_train_encoded = label.fit_transform(y_train)
    y_test_encoded = label.transform(y_test)

    # Convert text to feature vectors
    vectorizer = CountVectorizer(binary=True)  # BernoulliNB benefits from binary feature representation
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train_encoded, y_test_encoded

def Kfold(X,y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\nProcessing Fold {fold}")

        # Split data into train and test
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Preprocess data for each fold
        X_train_vec, X_test_vec, y_train_encoded, y_test_encoded = data_preprocessing(X_train, X_test,y_train, y_test)

        # Model training and prediction
        model = BernoulliNB()
        model.fit(X_train_vec, y_train_encoded)
        y_pred = model.predict(X_test_vec)

        # Calculate accuracy for the current fold
        acc = accuracy_score(y_test_encoded, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold}: Accuracy = {acc:.4f}")

    # Output mean accuracy across all folds
    print(f"\nMean Accuracy across 10 folds: {np.mean(accuracies):.4f}")

def main():
    # Load data
    X, y = load_data()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=32, stratify=y
    )

    # Preprocess
    X_train_vec, X_test_vec, y_train_encoded, y_test_encoded = data_preprocessing(
        X_train, X_test, y_train, y_test
    )

    # Train Bernoulli Naive Bayes model
    model = BernoulliNB()
    model.fit(X_train_vec, y_train_encoded)

    # Predict
    y_pred = model.predict(X_test_vec)

    # Evaluate
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=['ham', 'spam']))

    print("\nKFOLD Results:")
    Kfold(X,y)
if __name__ == "__main__":
    main()
