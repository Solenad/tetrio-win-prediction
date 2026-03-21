import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def plot_knn_boundary(k_value, weight_type, title, X_train, y_train):
    # 1. Pick two features for 2D plotting
    # Replace these with your top 2 features from your importance plot
    feature_a, feature_b = "pps", "apm"
    X_subset = X_train[[feature_a, feature_b]].values
    y_subset = y_train.values

    # 2. Fit a simplified model for visualization
    # We use a smaller sample (e.g., 500 points) so the plot isn't too cluttered
    idx = np.random.choice(len(X_subset), 500, replace=False)
    X_sample, y_sample = X_subset[idx], y_subset[idx]

    # Scale them specifically for this plot
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    clf = KNeighborsClassifier(n_neighbors=k_value, weights=weight_type)
    clf.fit(X_scaled, y_sample)

    # 3. Create a mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 4. Predict across the grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 5. Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(
        X_scaled[:, 0], X_scaled[:, 1], c=y_sample, edgecolors="k", cmap="RdBu", s=20
    )
    plt.title(title)
    plt.xlabel(f"Standardized {feature_a}")
    plt.ylabel(f"Standardized {feature_b}")
    plt.show()
