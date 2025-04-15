# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC


def PCA_custom():
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, cut_tree

    # Step 1: Load the dataset
    USArrests = get_rdataset('USArrests').data  # Built-in R dataset from statsmodels
    print(USArrests)
    print("Columns:", USArrests.columns)
    print("Mean:\n", USArrests.mean())
    print("Variance:\n", USArrests.var())

    # Step 2: Standardize the dataset (important before PCA)
    scaler = StandardScaler(with_mean=True, with_std=True)
    USArrests_scaled = scaler.fit_transform(USArrests)

    # Step 3: Perform PCA
    pcaUS = PCA()
    pcaUS.fit(USArrests_scaled)

    # Print PCA means (should be ~0 after scaling)
    print("Mean used by PCA:", pcaUS.mean_)

    # Get PCA scores (i.e., transformed data in new PC axes)
    scores = pcaUS.transform(USArrests_scaled)

    # Print PCA loadings (principal components)
    print("Principal Components (Loadings):\n", pcaUS.components_)

    # Step 4: Plot PCA Biplot (PC1 vs PC2)
    i, j = 0, 1  # Plotting the 1st and 2nd principal components

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # First biplot: original
    axes[0].scatter(scores[:, i], scores[:, j])
    axes[0].set_title("Original Orientation")
    axes[0].set_xlabel(f'PC{i+1}')
    axes[0].set_ylabel(f'PC{j+1}')
    for k in range(pcaUS.components_.shape[1]):
        axes[0].arrow(0, 0, pcaUS.components_[i, k], pcaUS.components_[j, k], color='r', head_width=0.05)
        axes[0].text(pcaUS.components_[i, k]*1.15, pcaUS.components_[j, k]*1.15, USArrests.columns[k], color='g')

    # Second biplot: scaled arrows
    axes[1].scatter(scores[:, i], -scores[:, j])
    axes[1].set_title("Flipped and Scaled")
    axes[1].set_xlabel(f'PC{i+1}')
    axes[1].set_ylabel(f'PC{j+1}')
    scale_arrow = 2
    for k in range(pcaUS.components_.shape[1]):
        axes[1].arrow(0, 0, scale_arrow * pcaUS.components_[i, k], scale_arrow * -pcaUS.components_[j, k], color='r', head_width=0.05)
        axes[1].text(scale_arrow * pcaUS.components_[i, k] * 1.1,
                     scale_arrow * pcaUS.components_[j, k] * -1.1,
                     USArrests.columns[k], color='g')

    plt.tight_layout()
    plt.show()

    # Step 6: Variance explained by each principal component
    print("Explained Variance:\n", pcaUS.explained_variance_)

    # Step 7: Scree Plot and Cumulative Variance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ticks = np.arange(1, pcaUS.n_components_ + 1)

    # Scree plot
    axes[0].plot(ticks, pcaUS.explained_variance_ratio_, marker='o')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Proportion of Variance Explained')
    axes[0].set_title("Scree Plot")
    axes[0].set_ylim([0, 1])
    axes[0].set_xticks(ticks)

    # Cumulative variance plot
    axes[1].plot(ticks, pcaUS.explained_variance_ratio_.cumsum(), marker='o')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Proportion of Variance Explained')
    axes[1].set_title("Cumulative Variance")
    axes[1].set_ylim([0, 1])
    axes[1].set_xticks(ticks)

    plt.tight_layout()
    plt.show()

    # Step 8: Illustration of cumsum
    a = np.array([1, 2, 8, -3])
    print("Cumulative sum of a:", np.cumsum(a))

def PCA_OnDataSet():
    def load_data():
        column_names = [f"Feature_{i}" for i in range(60)] + ["Label"] # sonar.csv has 60 features + 1 label column
        data = pd.read_csv("../sonar.csv", header=None, names=column_names)
        print(data.head())
        X = data.drop(columns=["Label"])  # Feature matrix
        y = data["Label"]  # Target labels
        return X,y

    def data_preprocessing(X_t, X_test, y_t, y_test):
        label = LabelEncoder()
        y_t_encoded = label.fit_transform(y_t)
        y_test_encoded = label.transform(y_test)

        scaler = StandardScaler()
        X_t = scaler.fit_transform(X_t)
        X_test = scaler.transform(X_test)

        return X_t, X_test, y_t_encoded, y_test_encoded

    def KFOLD(X, y, model):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        test_accuracies = []
        fold = 1
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            X_train_p, X_test_p, y_train_p, y_test_p = data_preprocessing(X_train, X_test, y_train, y_test)
            model.fit(X_train_p, y_train_p)
            y_pred = model.predict(X_test_p)
            acc = accuracy_score(y_test_p, y_pred)
            test_accuracies.append(acc)
            print(f"Fold {fold} - Test Accuracy: {acc:.4f}")
            fold += 1
        print(f"\nOverall Average Test Accuracy: {np.mean(test_accuracies):.4f}")

    X, y = load_data()
    pca = PCA(n_components=0.95)
    X_t = pca.fit_transform(X)
    model1 = LogisticRegression(max_iter=1000, random_state=32)
    model2 = SVC(random_state=42, max_iter=1000)
    KFOLD(X_t, y, model2)

if __name__ == "__main__":
    PCA_custom()
    PCA_OnDataSet()
