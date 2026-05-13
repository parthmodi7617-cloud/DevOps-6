"""
Generalized LDA (Linear Discriminant Analysis)
================================================
KNN + SVM with and without LDA dimensionality reduction
- Create your own synthetic dataset  →  edit CLASSES in CONFIG
- OR load your own CSV               →  set CSV_PATH + related vars in CONFIG
All plots from the original code are reproduced + enhanced.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
np.random.seed(42)

DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"

# ============================================================
#  CONFIG — EDIT THIS SECTION
# ============================================================

# ── OPTION A : Synthetic Gaussian clusters ───────────────────────────────────
# Each class is a cluster:
#   "center" : list of feature means  (length = number of features)
#   "std"    : spread (scalar or list matching length of center)
#   "n"      : number of samples
#   "label"  : class name (string)

CLASSES = [
    {"center": [1.0, 2.0, 0.5, 1.5], "std": 0.5, "n": 80,  "label": "Class_A"},
    {"center": [4.0, 1.0, 3.5, 2.0], "std": 0.5, "n": 80,  "label": "Class_B"},
    {"center": [2.5, 4.5, 1.0, 3.5], "std": 0.5, "n": 80,  "label": "Class_C"},
    {"center": [5.0, 3.0, 4.5, 0.5], "std": 0.5, "n": 80,  "label": "Class_D"},
]
FEATURE_NAMES = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]

# ── OPTION B : Load your own CSV ─────────────────────────────────────────────
# Steps to switch to CSV mode:
#   1. Un-comment the 3 lines below
#   2. Comment out the CLASSES list above
#   3. Set CSV_PATH to your file, TARGET_COLUMN to your label column
#      and FEATURE_COLS to the columns you want as features
#      (set FEATURE_COLS = None to use ALL columns except the target)
#
# CSV_PATH      = "your_file.csv"
# TARGET_COLUMN = "label"              # column that holds class labels
# FEATURE_COLS  = None                 # None = all columns except target
#                                      # or e.g. ["col1", "col2", "col3"]

# ── If your dataset has pixel/image data (like Fashion MNIST) ────────────────
# Un-comment the line below to normalise values to [0, 1]
# NORMALIZE = True   # divides every feature by 255

# ── Split settings ────────────────────────────────────────────────────────────
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# ── LDA settings ─────────────────────────────────────────────────────────────
# n_components must be <= (n_classes - 1)
# Set to None to auto-select (n_classes - 1)
LDA_N_COMPONENTS = None

# ── KNN settings ─────────────────────────────────────────────────────────────
KNN_K = 5

# ── SVM settings ─────────────────────────────────────────────────────────────
SVM_KERNEL = "linear"   # "linear" | "rbf" | "poly"
SVM_C      = 1.0

# ============================================================
#  END OF CONFIG
# ============================================================


# ── 1. Build / load dataset ───────────────────────────────────────────────────
def build_dataset():
    # ── CSV mode ──────────────────────────────────────────────────────────────
    # Un-comment this entire block when using a CSV:
    #
    # df = pd.read_csv(CSV_PATH)
    # feat_cols = (
    #     [c for c in df.columns if c != TARGET_COLUMN]
    #     if FEATURE_COLS is None
    #     else FEATURE_COLS
    # )
    # X = df[feat_cols].values.astype(float)
    # # Optional normalisation (e.g. pixel data)
    # # if NORMALIZE:
    # #     X = X / 255.0
    # le = LabelEncoder()
    # y  = le.fit_transform(df[TARGET_COLUMN].astype(str))
    # return X, y, feat_cols, list(le.classes_)

    # ── Synthetic cluster mode ────────────────────────────────────────────────
    X_parts, y_parts, labels = [], [], []
    for cid, cls in enumerate(CLASSES):
        n_feat = len(cls["center"])
        std    = cls["std"] if isinstance(cls["std"], list) else [cls["std"]] * n_feat
        pts    = np.column_stack([
            np.random.normal(cls["center"][f], std[f], cls["n"])
            for f in range(n_feat)
        ])
        X_parts.append(pts)
        y_parts.append(np.full(cls["n"], cid, dtype=int))
        labels.append(cls["label"])
    X = np.vstack(X_parts)
    y = np.hstack(y_parts)
    return X, y, FEATURE_NAMES, labels


X, y, feature_names, class_labels = build_dataset()
N_CLASSES  = len(np.unique(y))
N_FEATURES = X.shape[1]

# Auto LDA components
n_lda = min(N_CLASSES - 1, N_FEATURES) if LDA_N_COMPONENTS is None else LDA_N_COMPONENTS
n_lda = max(1, n_lda)

print("=" * 58)
print("  Dataset Summary")
print("=" * 58)
print(f"  Total samples  : {len(y)}")
print(f"  Features       : {N_FEATURES}  {feature_names}")
print(f"  Classes        : {N_CLASSES}  {class_labels}")
print(f"  LDA components : {n_lda}")
print("=" * 58)


# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"\n  Train: {X_train.shape}   Test: {X_test.shape}")


# ── helper: style axes ────────────────────────────────────────────────────────
def style_ax(ax, title="", xlabel="", ylabel="", grid=False):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title,   color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#ffffff33")
    if grid:
        ax.grid(True, color="#ffffff22", linestyle="--", linewidth=0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Sample scatter of raw features (first 2 dims)
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax_raw = plt.subplots(figsize=(8, 6))
fig1.patch.set_facecolor(DARK_BG)
palette = plt.cm.tab10(np.linspace(0, 0.6, N_CLASSES))
for cid in range(N_CLASSES):
    mask = y_train == cid
    ax_raw.scatter(X_train[mask, 0], X_train[mask, 1 % N_FEATURES],
                   color=palette[cid], alpha=0.75, edgecolors="white",
                   linewidths=0.3, s=45, label=class_labels[cid])
style_ax(ax_raw,
         f"Raw Dataset — {feature_names[0]} vs {feature_names[1 % N_FEATURES]}",
         feature_names[0], feature_names[1 % N_FEATURES], grid=True)
ax_raw.legend(facecolor=PANEL_BG, edgecolor="#ffffff33", labelcolor="white")
plt.tight_layout()
plt.savefig("lda_01_raw_data.png", dpi=150, facecolor=fig1.get_facecolor())
print("\n[INFO] Saved: lda_01_raw_data.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  LDA fit
# ══════════════════════════════════════════════════════════════════════════════
lda = LinearDiscriminantAnalysis(n_components=n_lda)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda  = lda.transform(X_test)
print(f"\n  After LDA — Train: {X_train_lda.shape}   Test: {X_test_lda.shape}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — LDA component scatter (first 2 components)
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax_lda = plt.subplots(figsize=(9, 7))
fig2.patch.set_facecolor(DARK_BG)
df_lda_plot = pd.DataFrame({
    "LD1"  : X_train_lda[:, 0],
    "LD2"  : X_train_lda[:, 1] if n_lda > 1 else np.zeros(len(X_train_lda)),
    "Class": [class_labels[c] for c in y_train]
})
for cid, lbl in enumerate(class_labels):
    sub = df_lda_plot[df_lda_plot["Class"] == lbl]
    ax_lda.scatter(sub["LD1"], sub["LD2"],
                   color=palette[cid], alpha=0.75,
                   edgecolors="white", linewidths=0.3,
                   s=50, label=lbl)
style_ax(ax_lda, "LDA — First Two Discriminant Components",
         "LDA Component 1", "LDA Component 2" if n_lda > 1 else "—", grid=True)
ax_lda.legend(facecolor=PANEL_BG, edgecolor="#ffffff33", labelcolor="white")
plt.tight_layout()
plt.savefig("lda_02_lda_scatter.png", dpi=150, facecolor=fig2.get_facecolor())
print("[INFO] Saved: lda_02_lda_scatter.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Explained Variance
# ══════════════════════════════════════════════════════════════════════════════
evr  = lda.explained_variance_ratio_
cumv = np.cumsum(evr)
comp = np.arange(1, len(evr) + 1)

fig3, ax_ev = plt.subplots(figsize=(8, 5))
fig3.patch.set_facecolor(DARK_BG)
ax_ev.bar(comp, evr, color="#5ca8e0", alpha=0.8,
          edgecolor="#ffffff22", label="Individual Variance")
ax_ev.step(comp, cumv, where="mid", color="#f0c040",
           marker="o", linewidth=2, label="Cumulative Variance")
for i, (iv, cv) in enumerate(zip(evr, cumv)):
    ax_ev.text(comp[i], iv + 0.005, f"{iv:.2%}",
               ha="center", color="white", fontsize=8)
style_ax(ax_ev, "LDA Explained Variance Ratio",
         "LDA Component", "Explained Variance Ratio", grid=True)
ax_ev.legend(facecolor=PANEL_BG, edgecolor="#ffffff33", labelcolor="white")
plt.tight_layout()
plt.savefig("lda_03_explained_variance.png", dpi=150, facecolor=fig3.get_facecolor())
print("[INFO] Saved: lda_03_explained_variance.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Train all 4 model variants
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Training models...")

# KNN — original
knn_orig = KNeighborsClassifier(n_neighbors=KNN_K)
knn_orig.fit(X_train, y_train)
y_pred_knn_orig = knn_orig.predict(X_test)
acc_knn_orig    = accuracy_score(y_test, y_pred_knn_orig)

# KNN — LDA
knn_lda = KNeighborsClassifier(n_neighbors=KNN_K)
knn_lda.fit(X_train_lda, y_train)
y_pred_knn_lda  = knn_lda.predict(X_test_lda)
acc_knn_lda     = accuracy_score(y_test, y_pred_knn_lda)

# SVM — original
svm_orig = SVC(kernel=SVM_KERNEL, C=SVM_C, random_state=RANDOM_SEED)
svm_orig.fit(X_train, y_train)
y_pred_svm_orig = svm_orig.predict(X_test)
acc_svm_orig    = accuracy_score(y_test, y_pred_svm_orig)

# SVM — LDA
svm_lda = SVC(kernel=SVM_KERNEL, C=SVM_C, random_state=RANDOM_SEED)
svm_lda.fit(X_train_lda, y_train)
y_pred_svm_lda  = svm_lda.predict(X_test_lda)
acc_svm_lda     = accuracy_score(y_test, y_pred_svm_lda)

results = {
    "KNN Original": (acc_knn_orig, y_pred_knn_orig),
    "KNN + LDA"   : (acc_knn_lda,  y_pred_knn_lda),
    "SVM Original": (acc_svm_orig, y_pred_svm_orig),
    "SVM + LDA"   : (acc_svm_lda,  y_pred_svm_lda),
}

print("\n" + "=" * 45)
print("  Model Accuracy Comparison")
print("=" * 45)
for name, (acc, _) in results.items():
    print(f"  {name:<18s}: {acc:.4f}  ({acc:.2%})")
print("=" * 45)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — LDA decision regions (2-D, first two components)
# ══════════════════════════════════════════════════════════════════════════════
if n_lda >= 2:
    x_min = X_train_lda[:, 0].min() - 1
    x_max = X_train_lda[:, 0].max() + 1
    y_min = X_train_lda[:, 1].min() - 1
    y_max = X_train_lda[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    fig6, (ax_knn_r, ax_svm_r) = plt.subplots(1, 2, figsize=(16, 6))
    fig6.patch.set_facecolor(DARK_BG)

    # Train 2-D classifiers just for plotting (first 2 LDA dims)
    knn_2d = KNeighborsClassifier(n_neighbors=KNN_K)
    knn_2d.fit(X_train_lda[:, :2], y_train)
    svm_2d = SVC(kernel=SVM_KERNEL, C=SVM_C, random_state=RANDOM_SEED)
    svm_2d.fit(X_train_lda[:, :2], y_train)

    for ax, clf, title in [
        (ax_knn_r, knn_2d, f"KNN (k={KNN_K}) Decision Regions — LDA Space"),
        (ax_svm_r, svm_2d, f"SVM ({SVM_KERNEL}) Decision Regions — LDA Space"),
    ]:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.22,
                    cmap=plt.cm.tab10,
                    levels=np.arange(-0.5, N_CLASSES))
        ax.contour(xx, yy, Z, colors="#ffffffaa", linewidths=1.0)
        for cid in range(N_CLASSES):
            mask = y_train == cid
            ax.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1],
                       color=palette[cid], alpha=0.8, edgecolors="white",
                       linewidths=0.3, s=40, label=class_labels[cid])
        style_ax(ax, title, "LDA Component 1", "LDA Component 2")
        ax.legend(facecolor=PANEL_BG, edgecolor="#ffffff33",
                  labelcolor="white", fontsize=8)

    fig6.suptitle("Decision Regions in LDA Space",
                  color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("lda_06_decision_regions.png", dpi=150,
                bbox_inches="tight", facecolor=fig6.get_facecolor())
    print("[INFO] Saved: lda_06_decision_regions.png")
    plt.show()
else:
    print("\n[INFO] Skipped decision regions plot — need n_lda >= 2")