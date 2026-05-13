import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier
)

import xgboost as xgb
import lightgbm as lgb


# ============================================================
# DATASET
# ============================================================

# [CHANGE HERE] If using your own dataset (like a CSV), load it directly using pandas:
# df = pd.read_csv("your_dataset.csv")
# 
# [CHANGE HERE] Drop the target column to get features (X) and isolate the target (y):
# X = df.drop("TargetColumnName", axis=1).values
# y = df["TargetColumnName"].values
# 
# [CHANGE HERE] Define your target class labels for the confusion matrix visualization:
# class_labels = ["ClassA", "ClassB", "ClassC"] 
# 
# After uncommenting and setting the above, you can remove or comment out the manual dataset generation below.

# [CHANGE HERE] The CLASSES and FEATURE_NAMES variables below are ONLY used for 
# generating the synthetic sample data. If you use a custom dataset (CSV), 
# you can completely delete or comment out everything down to "TRAIN TEST SPLIT".

CLASSES = [
    {"center": [4.5, 3.0, 1.5, 0.3], "std": 0.4, "n": 50, "label": "Setosa"},
    {"center": [5.9, 2.8, 4.2, 1.3], "std": 0.5, "n": 50, "label": "Versicolor"},
    {"center": [6.6, 3.0, 5.5, 2.0], "std": 0.5, "n": 50, "label": "Virginica"},
]

# [CHANGE HERE] Not needed if using a custom CSV. You can get feature names dynamically using:
# feature_names = df.drop("TargetColumnName", axis=1).columns.tolist()
FEATURE_NAMES = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth"
]

TEST_SIZE = 0.2
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# BUILD DATASET
# ============================================================

X_parts = []
y_parts = []

for cid, cls in enumerate(CLASSES):

    points = (
        np.random.randn(cls["n"], len(cls["center"])) * cls["std"]
        + np.array(cls["center"])
    )

    X_parts.append(points)
    y_parts.append(np.full(cls["n"], cid))

X = np.vstack(X_parts)
y = np.hstack(y_parts)

class_labels = [cls["label"] for cls in CLASSES]


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
# MODELS
# ============================================================

models = {

    "AdaBoost": AdaBoostClassifier(
        n_estimators=50,
        random_state=RANDOM_SEED
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED
    ),

    "XGBoost": xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        verbosity=0
    ),

    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        verbose=-1
    )
}


# ============================================================
# TRAINING
# ============================================================

results = []

print("=" * 60)
print("Boosting Model Comparison")
print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results.append({
        "name": name,
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "pred": test_pred
    })

    print(f"\n{name}")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================

df_results = pd.DataFrame({
    "Model": [r["name"] for r in results],
    "Train Accuracy": [round(r["train_acc"], 4) for r in results],
    "Test Accuracy": [round(r["test_acc"], 4) for r in results]
})

print("\n")
print(df_results)


# ============================================================
# BEST MODEL
# ============================================================

best = max(results, key=lambda x: x["test_acc"])

print("\n" + "=" * 60)
print(f"Best Model : {best['name']}")
print(f"Best Accuracy : {best['test_acc']:.4f}")
print("=" * 60)


# ============================================================
# ACCURACY COMPARISON GRAPH
# ============================================================

model_names = [r["name"] for r in results]
train_accs = [r["train_acc"] for r in results]
test_accs = [r["test_acc"] for r in results]

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, train_accs, width, label="Train Accuracy")
plt.bar(x + width/2, test_accs, width, label="Test Accuracy")

plt.xticks(x, model_names)

plt.ylabel("Accuracy")
plt.title("Boosting Models Accuracy Comparison")

plt.ylim(0, 1.1)

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# CONFUSION MATRIX FOR BEST MODEL
# ============================================================

cm = confusion_matrix(y_test, best["pred"])

fig, ax = plt.subplots(figsize=(5, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_labels
)

disp.plot(ax=ax, cmap="Blues", colorbar=False)

plt.title(f"Confusion Matrix - {best['name']}")

plt.tight_layout()
plt.show()