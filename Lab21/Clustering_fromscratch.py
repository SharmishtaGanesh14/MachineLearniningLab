import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def fit(self, X):
        # Randomly initialize centroids from the data points
        np.random.seed(42)
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Assign each point to the closest centroid
            self.labels = self._assign_clusters(X)

            # Compute new centroids from the means of the points in each cluster
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])

            # Check for convergence
            diff = np.linalg.norm(self.centroids - new_centroids)
            if diff < self.tolerance:
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        # Compute distances between each point and the centroids
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)

    def plot_clusters(self, X):
        plt.figure(figsize=(8, 5))
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
        plt.legend()
        plt.title('K-Means Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Fit K-Means
kmeans = KMeans(k=3)
kmeans.fit(X)

# Visualize
kmeans.plot_clusters(X)
