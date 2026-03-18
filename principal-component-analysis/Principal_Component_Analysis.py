import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)

# アルゴリズムの実装
def fit(X, n_components):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - mean) / std
    n = X_std.shape[0]
    cov_matrix = X_std.T @ X_std / n
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, sorted_indices[:n_components]]

    return components, mean, std, eigenvalues[sorted_indices]

def transform(X, components, mean, std):
    X_std = (X - mean) / std
    return X_std @ components

# 元のデータと主成分の可視化
components, mean, std, eigenvalues = fit(X, n_components=1)

plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
origin = mean
for comp, eigval in zip(components.T, eigenvalues[:1]):
    plt.annotate('', xy=origin + comp * np.sqrt(eigval) * 3, xytext=origin, arrowprops=dict(arrowstyle='->', color='red', lw=2))
plt.title("Original Data + Principal Component")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 次元削減後のデータの可視化
X_transformed = transform(X, components, mean, std)

plt.scatter(X_transformed, np.zeros_like(X_transformed), c=y, alpha=0.5)
plt.title("Transformed Data (1D)")
plt.xlabel("Principal Component 1")
plt.yticks([])
plt.show()

# 累積寄与率の可視化
components_all, _, _, eigenvalues_all = fit(X, n_components=2)
explained_variance_ratio = eigenvalues_all / np.sum(eigenvalues_all)
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.bar([1, 2], explained_variance_ratio, alpha=0.7, label='Explained Variance Ratio')
plt.plot([1, 2], cumulative_variance, marker='o', color='red', label='Cumulative')
plt.xticks([1, 2])
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio")
plt.legend()
plt.show()


# 元のデータと変換後のデータを比較
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
axes[0].set_title("Original Data (2D)")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

axes[1].scatter(X_transformed, np.zeros_like(X_transformed), c=y, alpha=0.5)
axes[1].set_title("Transformed Data (1D)")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_yticks([])

plt.tight_layout()
plt.show()