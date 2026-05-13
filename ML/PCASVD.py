import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from numpy.linalg import svd

# -----------------------------
# SMALL DATASET (OR LOAD YOUR OWN)
# -----------------------------

# To use an external dataset, comment out this mock dataframe 
# and uncomment the pd.read_csv line below:
# df = pd.read_csv('path_to_your_data.csv')

df = pd.DataFrame({
    "Feature1": [2,4,6,8,10,12,14,16],
    "Feature2": [1,3,5,7,9,11,13,15],
    "Feature3": [5,6,7,8,9,10,11,12],
    "Quality":  [3,4,4,5,5,6,7,8]
})

print("Dataset:\n")
print(df)

# Features
# If using an external dataset, replace 'Quality' with your target column name.
# If your dataset only has features and no target, use: X = df
X = df.drop('Quality', axis=1)

# -----------------------------------
# PCA
# -----------------------------------

pca = PCA()

X_pca = pca.fit_transform(X)

print("\n=== PCA METRICS ===")
print("Explained Variance:")
print(pca.explained_variance_[:5])

pca_var = pca.explained_variance_ratio_[:5]

# PCA Plot
plt.figure(figsize=(6,5))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df['Quality'], # Change 'Quality' to your target variable, or remove `c=...` if you have none
    cmap='viridis'
)

plt.colorbar(label='Quality') # Change label to your target name, or remove this line if no target

plt.title("PCA: First 2 Components")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()

# -----------------------------------
# SVD WITHOUT CENTERING
# -----------------------------------

X_uncentered = X.values

U, S, VT = svd(X_uncentered, full_matrices=False)

# Projection
X_svd = U[:, :2] * S[:2]

# Variance
svd_variance = (S**2) / (len(X_uncentered)-1)

svd_var1 = svd_variance[:5] / np.sum(svd_variance)

print("\n=== SVD (UNCENTERED) METRICS ===")
print("Explained Variance:")
print(svd_variance[:5])

# Plot
plt.figure(figsize=(6,5))

plt.scatter(
    X_svd[:,0],
    X_svd[:,1],
    c=df['Quality'],
    cmap='coolwarm'
)

plt.colorbar(label='Quality')

plt.title("SVD (Uncentered) First 2 Components")
plt.xlabel("SVD1")
plt.ylabel("SVD2")

plt.show()

# -----------------------------------
# SVD WITH CENTERING
# -----------------------------------

# Center Data
X_centered = X - X.mean(axis=0)

U2, S2, VT2 = svd(X_centered, full_matrices=False)

# Projection
X_svd_centered = U2[:, :2] * S2[:2]

# Variance
svd_variance2 = (S2**2) / (len(X_centered)-1)

svd_var2 = svd_variance2[:5] / np.sum(svd_variance2)

print("\n=== SVD (CENTERED) METRICS ===")
print("Explained Variance:")
print(svd_variance2[:5])

# Plot
plt.figure(figsize=(6,5))

plt.scatter(
    X_svd_centered[:,0],
    X_svd_centered[:,1],
    c=df['Quality'],
    cmap='plasma'
)

plt.colorbar(label='Quality')

plt.title("SVD (Centered → PCA Equivalent)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.show()

# -----------------------------------
# FINAL COMPARISON
# -----------------------------------

print("\n=== FINAL COMPARISON SUMMARY ===")

print("PCA Variance Ratio:")
print(pca_var)

print("\nUncentered SVD Variance Ratio:")
print(svd_var1)

print("\nCentered SVD Variance Ratio:")
print(svd_var2)

# -----------------------------------
# PCA vs Uncentered SVD
# -----------------------------------

plt.figure(figsize=(12,5))

# PCA
plt.subplot(1,2,1)

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df['Quality'],
    cmap='viridis'
)

plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

# SVD Uncentered
plt.subplot(1,2,2)

plt.scatter(
    X_svd[:,0],
    X_svd[:,1],
    c=df['Quality'],
    cmap='coolwarm'
)

plt.title("SVD (Uncentered)")
plt.xlabel("SVD1")
plt.ylabel("SVD2")

plt.suptitle("PCA vs Uncentered SVD")

plt.tight_layout()

plt.show()

# -----------------------------------
# PCA vs Centered SVD
# -----------------------------------

plt.figure(figsize=(12,5))

# PCA
plt.subplot(1,2,1)

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df['Quality'],
    cmap='viridis'
)

plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

# Centered SVD
plt.subplot(1,2,2)

plt.scatter(
    X_svd_centered[:,0],
    X_svd_centered[:,1],
    c=df['Quality'],
    cmap='plasma'
)

plt.title("SVD (Centered)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.suptitle("PCA vs Centered SVD")

plt.tight_layout()

plt.show()

# -----------------------------------
# EXPLAINED VARIANCE COMPARISON
# -----------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    pca_var,
    marker='o',
    label="PCA"
)

plt.plot(
    svd_var1,
    marker='x',
    label="SVD Non-Centered"
)

plt.plot(
    svd_var2,
    marker='s',
    label="SVD Centered"
)

plt.title("Explained Variance Comparison")

plt.xlabel("Component Index")
plt.ylabel("Variance Ratio")

plt.legend()

plt.grid(True)

plt.show()