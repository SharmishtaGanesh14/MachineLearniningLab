import numpy as np
from ISLP.cluster import compute_linkage
from matplotlib.pyplot import title
from scipy.cluster.hierarchy import cut_tree
from seaborn.matrix import dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.svm import SVC


def K_Means(data):
    fig, ax = plt.subplots(6, 4, figsize=(8, 8))  # Subplot arrangement
    fig.suptitle("K-Means Clustering for Different K", fontsize=16)
    ax = ax.flatten()  # Flattening the axes for easy access
    inertias = []
    ranges = list(range(1, 25))

    for i in range(1, 25):
        km = KMeans(n_clusters=i, random_state=34343, n_init=30)
        data_fitted = km.fit(data)  # Fitting KMeans
        ax[i - 1].scatter(data[:, 0], data[:, 1], c=km.labels_)
        ax[i - 1].set_title(f"K={i}")
        inertias.append(km.inertia_)
    plt.show()

    plt.figure(figsize=(8, 5))  # Elbow plot
    plt.plot(ranges, inertias, marker='o', linestyle='--')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (Within-cluster Sum of Squares)")
    plt.xticks(ranges)
    plt.grid(True)
    plt.show()


def load_data():
    column_names = [f"Feature_{i}" for i in range(60)] + ["Label"]  # sonar.csv has 60 features + 1 label column
    data = pd.read_csv("../sonar.csv", header=None, names=column_names)
    X = data.drop(columns=["Label"])  # Feature matrix
    y = data["Label"]  # Target labels
    return X, y


if __name__ == "__main__":
    # data = np.random.standard_normal(size=(50, 2))
    # data[:25, 0] = data[:25, 0] + 3
    # data[:25, 1] = data[:25, 1] - 4
    # K_Means(data)  # Test KMeans on random data

    X, y = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    model = KMeans(random_state=3526, n_init=40, n_clusters=2)
    model.fit(X_pca)
    labels = model.labels_
    # Get cluster centers (centroids)
    inertia = model.inertia_
    centroids = model.cluster_centers_
    X_pca_df = pd.DataFrame(X_pca)
    X_pca_df["Labels"] = labels
    X_pca_df["Org_labels"] = y.values
    print(X_pca_df.groupby('Labels').describe())
    print(X_pca_df.drop(columns='Labels').groupby('Org_labels').describe())

