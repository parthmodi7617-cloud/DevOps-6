"""
Generalized Non-Linear SVM (RBF Kernel)
=========================================
- Define your own dataset in the CONFIG section
- Supports any number of classes and cluster shapes
- Auto-applies PCA → 2D, scaling, SVM (RBF), metrics, and all plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
# X_raw = df.drop('target_column_name', axis=1).values
# y = df['target_column_name'].values
# CLASSES = [{"label": f"Class {str(c)}"} for c in np.unique(y)]
# N_CLASSES = len(CLASSES)
# COLORS = plt.cm.tab10(np.linspace(0, 0.6, N_CLASSES))


# ── DELETE FROM HERE TO REMOVE CUSTOM DATASET GENERATOR ────
# Each class is a cluster defined by:
#   "center" : [x, y, ...]  — cluster centre (any number of features)
#   "std"    : spread / noise (higher = more overlap)
#   "n"      : number of samples
#   "label"  : display name used in plots

CLASSES = [
    {"center": [ 3,  3,  1], "std": 1.0, "n": 150, "label": "Class A"},
    {"center": [-3, -3, -1], "std": 1.0, "n": 150, "label": "Class B"},
    {"center": [ 3, -3,  0], "std": 1.0, "n": 150, "label": "Class C"},
]

# PCA components to reduce to before SVM
#   2  → 2-D decision boundary plot (recommended)
#   None → skip PCA, use raw features (no boundary plot for >2 features)
PCA_COMPONENTS = 2

# Train/test split
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# RBF SVM hyper-parameters
KERNEL  = "rbf"     # "rbf" | "poly" | "sigmoid"
C_VALUE = 1.0       # regularisation (larger = tighter fit)
GAMMA   = "scale"   # "scale" | "auto" | float e.g. 0.1
DEGREE  = 3         # only used when kernel="poly"

# ============================================================
#  END OF CONFIG — no need to edit below this line
# ============================================================

np.random.seed(RANDOM_SEED)

# ── DELETE FROM HERE CONTINUED... ─────────────────────────────
N_CLASSES  = len(CLASSES)
COLORS     = plt.cm.tab10(np.linspace(0, 0.6, N_CLASSES))
AVG_MODE   = "binary" if N_CLASSES == 2 else "weighted"


# ── 1. Build dataset ──────────────────────────────────────────────────────────
X_parts, y_parts = [], []
for cid, cls in enumerate(CLASSES):
    pts = np.random.randn(cls["n"], len(cls["center"])) * cls["std"] \
          + np.array(cls["center"])
    X_parts.append(pts)
    y_parts.append(np.full(cls["n"], cid))

X_raw = np.vstack(X_parts)
y     = np.hstack(y_parts)
# ── END OF CUSTOM DATASET GENERATOR TO DELETE ───────────────

print("=" * 50)
print("  Dataset Summary")
print("=" * 50)
print(f"  Total samples  : {len(y)}")
print(f"  Features       : {X_raw.shape[1]}")
print(f"  Classes        : {N_CLASSES}")
for cid, cls in enumerate(CLASSES):
    print(f"    [{cid}] {cls['label']:12s} — {cls['n']} samples")
print("=" * 50)


# ── 2. Scale ──────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


# ── 3. Optional PCA ───────────────────────────────────────────────────────────
if PCA_COMPONENTS is not None and X_raw.shape[1] > PCA_COMPONENTS:
    pca  = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X_scaled)
    expl = pca.explained_variance_ratio_ * 100
    print(f"\n  PCA variance explained: "
          f"{expl[0]:.1f}% + {expl[1]:.1f}% = {sum(expl):.1f}%")
else:
    X_2d = X_scaled
    print("\n  PCA skipped (features already ≤ target dims or PCA_COMPONENTS=None)")


# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)


# ── 5. Raw dataset scatter (PCA space) ───────────────────────────────────────
fig0, ax0 = plt.subplots(figsize=(7, 5))
fig0.patch.set_facecolor("#0f1117")
ax0.set_facecolor("#1a1d27")
for cid, (cls, pts_raw, col) in enumerate(zip(CLASSES, X_parts, COLORS)):
    # project raw cluster through the same scaler+pca for plotting
    tmp = scaler.transform(pts_raw)
    tmp = pca.transform(tmp) if PCA_COMPONENTS else tmp
    ax0.scatter(tmp[:, 0], tmp[:, 1],
                color=col, alpha=0.75, edgecolors="white",
                linewidths=0.3, s=45, label=cls["label"])
ax0.set_title("Custom Dataset — PCA Space", color="white",
              fontsize=13, fontweight="bold")
ax0.set_xlabel("PCA Component 1", color="white")
ax0.set_ylabel("PCA Component 2", color="white")
ax0.tick_params(colors="white"); ax0.spines[:].set_color("#ffffff33")
ax0.legend(facecolor="#1a1d27", edgecolor="#ffffff33", labelcolor="white")
plt.tight_layout()
plt.savefig("nlsvm_01_raw_dataset.png", dpi=150, facecolor=fig0.get_facecolor())
print("\n[INFO] Saved: nlsvm_01_raw_dataset.png")
plt.show()


# ── 6. Train RBF SVM ─────────────────────────────────────────────────────────
model = SVC(kernel=KERNEL, C=C_VALUE, gamma=GAMMA,
            degree=DEGREE, random_state=RANDOM_SEED)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

train_acc  = accuracy_score(y_train, y_train_pred)
test_acc   = accuracy_score(y_test,  y_test_pred)
precision  = precision_score(y_test, y_test_pred, average=AVG_MODE, zero_division=0)
recall     = recall_score(y_test,    y_test_pred, average=AVG_MODE, zero_division=0)

print("\n" + "=" * 50)
print("  SVM (RBF) Performance")
print("=" * 50)
print(f"  Training Accuracy : {train_acc:.4f}  ({train_acc:.2%})")
print(f"  Testing  Accuracy : {test_acc:.4f}  ({test_acc:.2%})")
print(f"  Precision         : {precision:.4f}  ({precision:.2%})")
print(f"  Recall            : {recall:.4f}  ({recall:.2%})")
print("=" * 50)


# ── 7. Decision boundary helper ───────────────────────────────────────────────
def plot_decision_boundary(ax, X, y, model, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.set_facecolor("#1a1d27")
    ax.contourf(xx, yy, Z, alpha=0.25,
                cmap=plt.cm.RdYlBu,
                levels=np.arange(-0.5, N_CLASSES))
    ax.contour(xx, yy, Z, colors="#ffffffaa", linewidths=1.2)

    for cid, (cls, col) in enumerate(zip(CLASSES, COLORS)):
        mask = y == cid
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=col, alpha=0.80, edgecolors="white",
                   linewidths=0.3, s=40, label=cls["label"])

    # mark support vectors
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=100,
               facecolors="none", edgecolors="yellow",
               linewidths=1.2, label="Support Vectors")

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("PCA Component 1", color="white")
    ax.set_ylabel("PCA Component 2", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#ffffff33")
    ax.legend(facecolor="#1a1d27", edgecolor="#ffffff33",
              labelcolor="white", fontsize=8)


# ── 8. Results figure: confusion matrix + 2 boundary panels ──────────────────
fig = plt.figure(figsize=(20, 6))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                        left=0.05, right=0.97, top=0.88, bottom=0.12)

# Panel A — Confusion matrix
ax_cm = fig.add_subplot(gs[0])
cm    = confusion_matrix(y_test, y_test_pred)
labels = [cls["label"] for cls in CLASSES]
disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title(
    f"Confusion Matrix — {KERNEL.upper()} SVM\n"
    f"Train: {train_acc:.2%}  Test: {test_acc:.2%}  "
    f"Prec: {precision:.2%}  Rec: {recall:.2%}",
    color="white", fontsize=10, fontweight="bold"
)
ax_cm.set_facecolor("#1a1d27")
ax_cm.tick_params(colors="white")
ax_cm.xaxis.label.set_color("white"); ax_cm.yaxis.label.set_color("white")
for text in ax_cm.texts:
    text.set_color("black")

# Panel B — Training boundary
ax_tr = fig.add_subplot(gs[1])
plot_decision_boundary(ax_tr, X_train, y_train, model,
                       title=f"Training Set — {KERNEL.upper()} Kernel Decision Boundary")

# Panel C — Test boundary
ax_te = fig.add_subplot(gs[2])
plot_decision_boundary(ax_te, X_test, y_test, model,
                       title=f"Test Set — {KERNEL.upper()} Kernel Decision Boundary")

fig.suptitle(f"Non-Linear SVM ({KERNEL.upper()} Kernel) — Full Analysis",
             color="white", fontsize=15, fontweight="bold", y=0.97)

plt.savefig("nlsvm_02_results.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("[INFO] Saved: nlsvm_02_results.png")
plt.show()
