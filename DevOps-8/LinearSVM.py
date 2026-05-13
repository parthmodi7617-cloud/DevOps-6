"""
Generalized Linear SVM Classifier
====================================
- Define your own dataset in the CONFIG section below
- The rest runs automatically: scaling, train/test split, SVM, metrics, plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
#  CONFIG — EDIT THIS SECTION TO CREATE YOUR OWN DATASET
# ============================================================
# --- CHANGES FOR CUSTOM DATASET START HERE ---
# To use your own external dataset (e.g., from a CSV file), 
# uncomment the lines below and delete the dummy dataset generation block below it:
# 
# import pandas as pd
# df = pd.read_csv("your_dataset.csv")
# X = df[['feature1', 'feature2']].values # Adjust features as needed
# y = df['target_label'].values           # Adjust target as needed
# CLASSES = [
#     {"label": "Class A"},  # define based on how many classes are in your test dataset
#     {"label": "Class B"} 
# ]
# n_classes = len(np.unique(y))
# colors = plt.cm.tab10(np.linspace(0, 0.4, n_classes))

# ── DELETE FROM HERE TO REMOVE CUSTOM DATASET GENERATOR ────
# Each class is defined by:
#   "center"  : [mean_x, mean_y]  — where the cluster is placed
#   "std"     : spread of the cluster (higher = more overlap)
#   "n"       : number of samples in this class
#   "label"   : display name for legend/plot

CLASSES = [
    {"center": [ 2,  2], "std": 1.0, "n": 100, "label": "Class A"},
    {"center": [-2, -2], "std": 1.0, "n": 100, "label": "Class B"},
]


# Train/test split ratio  (0.2 = 20% test)
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

# LinearSVC settings
MAX_ITER     = 5000   # increase if model doesn't converge
C_VALUE      = 1.0    # regularisation strength (smaller = wider margin)

# ============================================================
#  END OF CONFIG — no need to edit below this line
# ============================================================

np.random.seed(RANDOM_SEED)


# ── 1. Build dataset ─────────────────────────────────────────────────────────
X_parts, y_parts = [], []
for class_id, cls in enumerate(CLASSES):
    pts = np.random.randn(cls["n"], 2) * cls["std"] + np.array(cls["center"])
    X_parts.append(pts)
    y_parts.append(np.full(cls["n"], class_id))

X = np.vstack(X_parts)
y = np.hstack(y_parts)

n_classes = len(CLASSES)
colors    = plt.cm.tab10(np.linspace(0, 0.4, n_classes))
# ── END OF CUSTOM DATASET GENERATOR TO DELETE ───────────────


# ── 2. Raw dataset scatter plot ───────────────────────────────────────────────
fig0, ax0 = plt.subplots(figsize=(7, 5))
fig0.patch.set_facecolor("#0f1117")
ax0.set_facecolor("#1a1d27")
for cid, (cls, pts, col) in enumerate(zip(CLASSES, X_parts, colors)):
    ax0.scatter(pts[:, 0], pts[:, 1],
                color=col, alpha=0.75, edgecolors="white",
                linewidths=0.3, s=45, label=cls["label"])
ax0.set_title("Custom Dataset (Raw)", color="white", fontsize=13, fontweight="bold")
ax0.set_xlabel("Feature 1", color="white"); ax0.set_ylabel("Feature 2", color="white")
ax0.tick_params(colors="white"); ax0.spines[:].set_color("#ffffff33")
ax0.legend(facecolor="#1a1d27", edgecolor="#ffffff33", labelcolor="white")
plt.tight_layout()
plt.savefig("svm_01_raw_dataset.png", dpi=150, facecolor=fig0.get_facecolor())
print("[INFO] Saved: svm_01_raw_dataset.png")
plt.show()


# ── 3. Train / test split + scaling ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ── 4. Train LinearSVC ───────────────────────────────────────────────────────
model = LinearSVC(C=C_VALUE, max_iter=MAX_ITER, random_state=RANDOM_SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ── 5. Metrics ───────────────────────────────────────────────────────────────
avg_mode = "binary" if n_classes == 2 else "weighted"
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average=avg_mode, zero_division=0)
rec  = recall_score(y_test, y_pred, average=avg_mode, zero_division=0)

print("\n" + "=" * 45)
print("  SVM Performance Metrics")
print("=" * 45)
print(f"  Accuracy  : {acc:.4f}  ({acc:.2%})")
print(f"  Precision : {prec:.4f}  ({prec:.2%})")
print(f"  Recall    : {rec:.4f}  ({rec:.2%})")
print("=" * 45)


# ── 6. Combined results figure: confusion matrix + decision boundary ──────────
fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                        left=0.06, right=0.97, top=0.88, bottom=0.12)

# ── Panel A: Confusion Matrix ─────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[0])
cm    = confusion_matrix(y_test, y_pred)
class_labels = [cls["label"] for cls in CLASSES]
disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title(
    f"Confusion Matrix — Linear SVM\n"
    f"Acc: {acc:.2%}  |  Prec: {prec:.2%}  |  Recall: {rec:.2%}",
    color="white", fontsize=11, fontweight="bold"
)
ax_cm.set_facecolor("#1a1d27")
ax_cm.tick_params(colors="white")
ax_cm.xaxis.label.set_color("white"); ax_cm.yaxis.label.set_color("white")
for text in ax_cm.texts:
    text.set_color("black")

# ── Panel B: Decision Boundary (scaled space) ─────────────────────────────────
ax_db = fig.add_subplot(gs[1])
ax_db.set_facecolor("#1a1d27")

# background mesh
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax_db.contourf(xx, yy, Z, alpha=0.18,
               cmap=plt.cm.RdYlBu, levels=np.arange(-0.5, n_classes))
ax_db.contour(xx, yy, Z, colors="#ffffff", linewidths=1.5, alpha=0.8)

# training points
for cid, (cls, col) in enumerate(zip(CLASSES, colors)):
    mask = y_train == cid
    ax_db.scatter(X_train[mask, 0], X_train[mask, 1],
                  color=col, alpha=0.75, edgecolors="white",
                  linewidths=0.3, s=40, label=cls["label"])

# decision boundary line (only well-defined for binary)
if n_classes == 2:
    w = model.coef_[0]
    b = model.intercept_[0]
    x_vals = np.linspace(x_min, x_max, 200)
    if abs(w[1]) > 1e-6:
        y_vals       = -(w[0] * x_vals + b) / w[1]
        y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
        y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]
        ax_db.plot(x_vals, y_vals,       "w-",  lw=2,   label="Decision boundary")
        ax_db.plot(x_vals, y_margin_pos, "w--", lw=1.2, alpha=0.55, label="Margin")
        ax_db.plot(x_vals, y_margin_neg, "w--", lw=1.2, alpha=0.55)

ax_db.set_xlim(x_min, x_max); ax_db.set_ylim(y_min, y_max)
ax_db.set_title("Linear SVM — Decision Boundary (Scaled)",
                color="white", fontsize=11, fontweight="bold")
ax_db.set_xlabel("Feature 1 (Scaled)", color="white")
ax_db.set_ylabel("Feature 2 (Scaled)", color="white")
ax_db.tick_params(colors="white"); ax_db.spines[:].set_color("#ffffff33")
ax_db.legend(facecolor="#1a1d27", edgecolor="#ffffff33",
             labelcolor="white", fontsize=8)

fig.suptitle("Linear SVM — Full Analysis", color="white",
             fontsize=15, fontweight="bold", y=0.97)

plt.show()
