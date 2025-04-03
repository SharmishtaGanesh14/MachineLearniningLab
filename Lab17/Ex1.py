import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay


# Define the transformation function
def Transform(Xd):
    return np.array([Xd[0] ** 2, np.sqrt(2) * Xd[0] * Xd[1], Xd[1] ** 2])


# Function to plot 2D data
def plot_2D_data(X, y):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={"Blue": "blue", "Red": "red"})
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Original Data (2D)")
    plt.show()


# Function to plot decision boundary in 2D
def plot_decision_boundary_2D(X, y, kernel="linear", ax=None, long_title=True, support_vectors=True):
    # Train the SVC
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)

    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    # x_min, x_max, y_min, y_max = -3, 3, -3, 3
    # ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")

    if support_vectors:
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if long_title:
        ax.set_title(f"Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(kernel)

    if ax is None:
        plt.show()


# Function to plot transformed data in 3D
def plot_3D_transformed_data(X_transformed, y):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for label in np.unique(y):
        mask = y == label
        ax.scatter(
            X_transformed[mask, 0],
            X_transformed[mask, 1],
            X_transformed[mask, 2],
            label=label,
            color="blue" if label == "Blue" else "red"
        )

    ax.set_xlabel("x1^2")
    ax.set_ylabel("√2 * x1 * x2")
    ax.set_zlabel("x2^2")
    ax.set_title("Transformed Data (3D)")
    ax.legend()
    plt.show()


# Function to plot decision boundary in 3D
def plot_3D_decision_boundary(X_transformed, y):
    clf = svm.SVC(kernel="linear").fit(X_transformed, y)

    x_range = np.linspace(X_transformed[:, 0].min(), X_transformed[:, 0].max(), 30)
    y_range = np.linspace(X_transformed[:, 1].min(), X_transformed[:, 1].max(), 30)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = -(
                    clf.coef_[0, 0] * xx[i, j] + clf.coef_[0, 1] * yy[i, j] + clf.intercept_
            ) / clf.coef_[0, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label in np.unique(y):
        mask = y == label
        ax.scatter(
            X_transformed[mask, 0],
            X_transformed[mask, 1],
            X_transformed[mask, 2],
            label=label,
            edgecolors="k",
        )
        # Plot Support Vectors
    support_vectors = clf.support_vectors_
    ax.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        support_vectors[:, 2],
        c='gray',
        marker='o',
        edgecolors='yellow',
        s=100,
        label='Support Vectors'
    )

    ax.plot_surface(xx, yy, zz, color="gray", alpha=0.5)
    ax.legend()
    plt.show()


# Kernel function
def kernel_function(a, b):
    return a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2


# Main function
def main():
    data = np.array([
        [1, 13, "Blue"], [1, 18, "Blue"], [2, 9, "Blue"], [3, 6, "Blue"],
        [6, 3, "Blue"], [9, 2, "Blue"], [13, 1, "Blue"], [18, 1, "Blue"],
        [3, 15, "Red"], [6, 6, "Red"], [6, 11, "Red"], [9, 5, "Red"],
        [10, 10, "Red"], [11, 5, "Red"], [12, 6, "Red"], [16, 3, "Red"]
    ])

    X = np.array([[int(row[0]), int(row[1])] for row in data])
    y = np.array(data[:, 2])
    plot_2D_data(X, y)
    y_numeric = np.where(y == "Blue", 0, 1)
    plot_decision_boundary_2D(X, y_numeric, kernel="linear")
    X_transformed = np.array([Transform(x) for x in X])
    plot_3D_transformed_data(X_transformed, y)
    plot_3D_decision_boundary(X_transformed, y)

    x1, x2 = np.array([3, 6]), np.array([10, 10])
    print("Dot product before transformation:", np.dot(x1, x2))
    print("Dot product in transformed space:", np.dot(Transform(x1), Transform(x2)))
    print(f"Result of applying kernel function:", kernel_function(x1, x2))

    # Interpretation of the Dot Product Result:
    # If the dot product increases after transformation, it suggests the vectors are more aligned in the new space.
    # Kernel methods (like this quadratic mapping) enhance similarity between points that originally seemed different in 2D.
    # This is the basis of SVM with kernel tricks, where nonlinear data is made linearly separable in higher dimensions.


if __name__ == "__main__":
    main()

