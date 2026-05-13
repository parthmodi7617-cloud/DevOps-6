import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

# ============================================================
# DATASET
# ============================================================
# --- CHANGES FOR CUSTOM DATASET START HERE ---
# To use your own dataset, you can delete this synthetic data generation
# and load your own data. For example:
# df = pd.read_csv("your_dataset.csv")
# X = df.drop("target_column", axis=1).values
# y = df["target_column"].values
# FEATURE_NAMES = df.columns.drop("target_column").tolist()
# CLASS_NAMES = np.unique(y).astype(str).tolist()

CLASSES = [
    {"center": [1.0, 2.0, 0.5, 1.5], "std": 0.5, "n": 80},
    {"center": [4.0, 1.0, 3.5, 2.0], "std": 0.5, "n": 80},
    {"center": [2.5, 4.5, 1.0, 3.5], "std": 0.5, "n": 80},
    {"center": [5.0, 3.0, 4.5, 0.5], "std": 0.5, "n": 80},
]

FEATURE_NAMES = [
    "Feature_1",
    "Feature_2",
    "Feature_3",
    "Feature_4"
]

CLASS_NAMES = [
    "Class_A",
    "Class_B",
    "Class_C",
    "Class_D"
]

TEST_SIZE = 0.2
RANDOM_SEED = 42

# ============================================================
# BUILD DATASET
# ============================================================

X_parts = []
y_parts = []

for cid, cls in enumerate(CLASSES):

    points = np.column_stack([
        np.random.normal(cls["center"][i], cls["std"], cls["n"])
        for i in range(4)
    ])

    X_parts.append(points)

    y_parts.append(np.full(cls["n"], cid))

X = np.vstack(X_parts)
y = np.hstack(y_parts)
# --- CHANGES FOR CUSTOM DATASET END HERE ---

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

# ============================================================
# RAW DATA VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

for cid in range(len(CLASS_NAMES)):

    mask = y_train == cid

    plt.scatter(
        X_train[mask, 0],
        X_train[mask, 1],
        label=CLASS_NAMES[cid]
    )

plt.xlabel(FEATURE_NAMES[0])
plt.ylabel(FEATURE_NAMES[1])

plt.title("Original Dataset")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# APPLY LDA
# ============================================================

lda = LinearDiscriminantAnalysis(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)

X_test_lda = lda.transform(X_test)

# ============================================================
# LDA VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

for cid in range(len(CLASS_NAMES)):

    mask = y_train == cid

    plt.scatter(
        X_train_lda[mask, 0],
        X_train_lda[mask, 1],
        label=CLASS_NAMES[cid]
    )

plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")

plt.title("LDA Transformed Data")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# EXPLAINED VARIANCE GRAPH
# ============================================================

explained_variance = lda.explained_variance_ratio_

components = np.arange(1, len(explained_variance) + 1)

plt.figure(figsize=(7, 5))

plt.bar(
    components,
    explained_variance,
    label="Individual Variance"
)

plt.plot(
    components,
    np.cumsum(explained_variance),
    marker='o',
    label="Cumulative Variance"
)

plt.xlabel("LDA Components")
plt.ylabel("Explained Variance Ratio")

plt.title("LDA Explained Variance")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# SIMPLE CLASSIFICATION USING LDA FEATURES
# ============================================================

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_lda, y_train)

predictions = model.predict(X_test_lda)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy after applying LDA:", round(accuracy, 4))

# ============================================================
# DECISION REGIONS
# ============================================================

x_min = X_train_lda[:, 0].min() - 1
x_max = X_train_lda[:, 0].max() + 1

y_min = X_train_lda[:, 1].min() - 1
y_max = X_train_lda[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

model_2d = KNeighborsClassifier(n_neighbors=5)

model_2d.fit(X_train_lda[:, :2], y_train)

Z = model_2d.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')

for cid in range(len(CLASS_NAMES)):

    mask = y_train == cid

    plt.scatter(
        X_train_lda[mask, 0],
        X_train_lda[mask, 1],
        label=CLASS_NAMES[cid]
    )

plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")

plt.title("LDA Decision Regions")

plt.legend()

plt.show()