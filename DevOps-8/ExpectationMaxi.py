import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

np.random.seed(42)

# ============================================================
# DATASET
# ============================================================

CLASSES = [
    {"center": [20, 80], "std": [8, 10], "n": 40},
    {"center": [20, 20], "std": [8, 10], "n": 40},
    {"center": [60, 50], "std": [8, 8],  "n": 40},
    {"center": [90, 80], "std": [8, 10], "n": 40},
    {"center": [90, 20], "std": [8, 10], "n": 40},
]

FEATURE_NAMES = [
    "Annual Income",
    "Spending Score"
]

N_CLUSTERS = 5
EM_ITERATIONS = 15

# ============================================================
# BUILD DATASET OR LOAD EXTERNAL DATASET
# ============================================================
# TO USE AN EXTERNAL DATASET (e.g., a CSV file), MAKE THE FOLLOWING CHANGES:
# 1. Comment out or remove the custom data generation loop below 
#    (from `parts = []` down to `df = pd.DataFrame(...)`).
# 2. Load your actual dataset using pandas:
#    df = pd.read_csv('your_dataset.csv')
# 3. Keep only the features you want to cluster on:
#    df = df[['Your_Feature_1', 'Your_Feature_2']]
# 4. Update the FEATURE_NAMES variable above to match your chosen columns 
#    so the graphs display the correct axes labels.
# ============================================================

# The code below generates a synthetic dataset with 5 clusters. If you load your own dataset, you can remove or comment out this entire section.

parts = []

for cls in CLASSES:

    points = np.column_stack([
        np.random.normal(cls["center"][i], cls["std"][i], cls["n"])
        for i in range(2)
    ])

    parts.append(points)

data = np.vstack(parts)

df = pd.DataFrame(data, columns=FEATURE_NAMES)

print("=" * 50)
print("Dataset Summary")
print("=" * 50)
print(df.head())

# ============================================================
# FEATURE SCALING
# ============================================================

X = df.values

scaler = StandardScaler()

scaled_data = scaler.fit_transform(X)

# ============================================================
# ELBOW METHOD
# ============================================================

wcss = []

k_range = range(1, 11)

for k in k_range:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(scaled_data)

    wcss.append(km.inertia_)

# ============================================================
# ELBOW GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(k_range, wcss, marker='o')

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.title("Elbow Method for Optimal Clusters")

plt.grid(True)

plt.show()

# ============================================================
# KMEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(scaled_data)

# ============================================================
# GMM CLUSTERING
# ============================================================

gmm = GaussianMixture(
    n_components=N_CLUSTERS,
    random_state=42
)

gmm_labels = gmm.fit_predict(scaled_data)

# ============================================================
# SILHOUETTE SCORES
# ============================================================

km_score = silhouette_score(scaled_data, kmeans_labels)

gmm_score = silhouette_score(scaled_data, gmm_labels)

print("\n" + "=" * 50)
print("Silhouette Scores")
print("=" * 50)

print("KMeans :", round(km_score, 4))
print("GMM    :", round(gmm_score, 4))

# ============================================================
# CLUSTER VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# KMeans Plot
axes[0].scatter(
    df[FEATURE_NAMES[0]],
    df[FEATURE_NAMES[1]],
    c=kmeans_labels,
    cmap='tab10'
)

axes[0].set_title("KMeans Clustering")
axes[0].set_xlabel(FEATURE_NAMES[0])
axes[0].set_ylabel(FEATURE_NAMES[1])

# GMM Plot
axes[1].scatter(
    df[FEATURE_NAMES[0]],
    df[FEATURE_NAMES[1]],
    c=gmm_labels,
    cmap='tab10'
)

axes[1].set_title("GMM Clustering")
axes[1].set_xlabel(FEATURE_NAMES[0])
axes[1].set_ylabel(FEATURE_NAMES[1])

plt.tight_layout()

plt.show()

# ============================================================
# EM CONVERGENCE GRAPH
# ============================================================

log_likelihoods = []

gmm_step = GaussianMixture(
    n_components=N_CLUSTERS,
    max_iter=1,
    warm_start=True,
    random_state=42
)

for i in range(EM_ITERATIONS):

    gmm_step.fit(scaled_data)

    ll = gmm_step.score(scaled_data) * len(scaled_data)

    log_likelihoods.append(ll)

# ============================================================
# EM CONVERGENCE PLOT
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    range(1, EM_ITERATIONS + 1),
    log_likelihoods,
    marker='o'
)

plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")

plt.title("EM Convergence Graph")

plt.grid(True)

plt.show()