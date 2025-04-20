import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_and_preprocess_data():
    nci = load_data("NCI60")
    X = nci['data']
    y = nci["labels"]["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.astype("category")


def reduce_with_pca(X_scaled, n_components=50):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca


def reduce_with_hierarchical_clustering(X_scaled, n_clusters=50):
    Z = linkage(X_scaled.T, method='ward')  # clustering genes
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    X_clustered = np.zeros((X_scaled.shape[0], n_clusters))
    for i in range(1, n_clusters + 1):
        X_clustered[:, i - 1] = X_scaled[:, clusters == i].mean(axis=1)
    return X_clustered


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


def plot_pca(X_pca, y_labels):
    # y_numeric = y_labels.cat.codes  # Convert categories to numeric codes
    # categories = y_labels.cat.categories

    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(
    #     X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='rainbow', edgecolor='k', alpha=0.8
    # )

    # # Create custom legend
    # handles = [
    #     plt.Line2D([], [], marker='o', linestyle='', color=scatter.cmap(scatter.norm(i)),
    #                label=cat, markersize=8)
    #     for i, cat in enumerate(categories)
    # ]
    # plt.legend(handles=handles, title="Cancer Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.title("PCA - NCI60 Gene Expression (PC1 vs PC2)")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    y_numeric = y_labels.cat.codes  # Convert categories to numeric codes
    categories = y_labels.cat.categories

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='rainbow', edgecolor='k', alpha=0.8
    )

    # Create custom legend
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=scatter.cmap(scatter.norm(i)),
                   label=cat, markersize=8)
        for i, cat in enumerate(categories)
    ]
    plt.legend(handles=handles, title="Cancer Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("PCA - NCI60 Gene Expression (PC1 vs PC2)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    y_numeric = y_labels.cat.codes
    categories = y_labels.cat.categories

    plt.figure(figsize=(8, 6))

    for i, category in enumerate(categories):
        idx = y_numeric == i
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                    label=category, alpha=0.8, edgecolor='k')

    plt.title("PCA - NCI60 Gene Expression (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cancer Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    X_scaled, y = load_and_preprocess_data()

    # PCA
    X_pca = reduce_with_pca(X_scaled, n_components=50)
    acc_pca = train_and_evaluate(X_pca, y)

    # Hierarchical Clustering
    X_clust = reduce_with_hierarchical_clustering(X_scaled, n_clusters=50)
    acc_clust = train_and_evaluate(X_clust, y)

    # Results
    print("\n--- Classification Accuracy ---")
    print(f"PCA-based Logistic Regression: {acc_pca:.4f}")
    print(f"Hierarchical Clustering-based Logistic Regression: {acc_clust:.4f}")

    # Visualization
    plot_pca(X_pca, y)


if __name__ == "__main__":
    main()
