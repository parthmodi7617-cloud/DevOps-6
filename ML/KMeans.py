import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Dataset
data = pd.DataFrame({
    "Annual_Income": [
        15,16,17,18,19,
        40,42,43,45,46,
        70,72,74,75,78
    ],

    "Spending_Score": [
        39,40,42,43,45,
        60,61,63,65,66,
        15,18,20,22,25
    ]
})

print("Dataset:\n")
print(data)

# Feature Scaling
scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

# Elbow Method
wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_data)

    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()

# Apply KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

# Add cluster column
data['Cluster'] = clusters

print("\nClustered Data:\n")
print(data)

# Centroids
print("\nCentroids:")
print(kmeans.cluster_centers_)

# Scatter Plot
plt.scatter(
    data['Annual_Income'],
    data['Spending_Score'],
    c=data['Cluster']
)

# Plot centroids
centroids = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    s=300,
    marker='X'
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")

plt.show()