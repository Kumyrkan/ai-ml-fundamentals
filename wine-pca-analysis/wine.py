import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "wine.data")

data = np.genfromtxt(file_path, delimiter=",")

y = data[:, 0].astype(int)
X = data[:, 1:].astype(float)


# ---- standardization ----
def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


X_std = standardize(X)


# ---- PCA ----
def pca(X, k):
    X_centered = X - X.mean(axis=0)

    cov = np.cov(X_centered, rowvar=False, ddof=1)

    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    explained_variance_ratio = eigvals / eigvals.sum()

    components = eigvecs[:, :k]
    projection = X_centered @ components

    return projection, explained_variance_ratio


# ---- PCA full ----
proj13, var_ratio = pca(X_std, 13)

print("explained_variance_ratio:")
print(" ".join(f"{x:.6f}" for x in var_ratio))

print("sum:")
print(f"{np.sum(var_ratio):.6f}")


# ---- cumulative variance ----
cum_var = np.cumsum(var_ratio)

print("\ncumulative variance:")
print(" ".join(f"{x:.6f}" for x in cum_var))

k_80 = np.argmax(cum_var >= 0.8) + 1
print("\nsmallest k for 80% variance:")
print(k_80)


# ---- PCA k=2 ----
proj2, _ = pca(X_std, 2)

print("\nprojection shape:", proj2.shape)

classes = np.unique(y)

for c in classes:
    m = proj2[y == c].mean(axis=0)
    print(f"class {c} mean: [{m[0]:.4f} {m[1]:.4f}]")