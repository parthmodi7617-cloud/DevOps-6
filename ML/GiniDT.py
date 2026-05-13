"""
Generalized Decision Tree with Gini Index
==========================================
- Define your own dataset (features + target) in the CONFIG section
- The rest runs automatically: encoding, train/test split, fitting, accuracy, plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ============================================================
#  CONFIG — EDIT THIS SECTION TO CREATE YOUR OWN DATASET
# ============================================================

# 1. Your raw data as a dictionary.
#    Each key = a column name, each value = a list of values.
#    Columns can be text (categorical) or numbers — both are handled.

RAW_DATA = {
    "Age":          ["Youth", "Youth", "Middle", "Senior", "Senior",
                     "Senior", "Middle", "Youth", "Youth", "Senior",
                     "Youth", "Middle", "Middle", "Senior", "Youth"],
    "Income":       ["High", "High", "High", "Medium", "Low",
                     "Low", "Low", "Medium", "Low", "Medium",
                     "Medium", "Medium", "High", "Medium", "Low"],
    "Student":      ["No", "No", "No", "No", "Yes",
                     "Yes", "Yes", "No", "Yes", "Yes",
                     "Yes", "No", "Yes", "No", "Yes"],
    "CreditRating": ["Fair", "Excellent", "Fair", "Fair", "Fair",
                     "Excellent", "Excellent", "Fair", "Fair", "Fair",
                     "Excellent", "Excellent", "Fair", "Excellent", "Fair"],
    "Buy_Product":  ["No", "No", "Yes", "Yes", "Yes",
                     "No", "Yes", "No", "Yes", "Yes",
                     "Yes", "Yes", "Yes", "No", "Yes"],
}

# 2. Name of the target column (what you want to predict)
#    (CHANGE THIS to the exact name of the target column in your new dataset)
TARGET_COLUMN = "Buy_Product"

# 3. Train/test split point (number of rows to use for training)
#    Remaining rows become the test set.
#    (CHANGE THIS to a valid number based on your new dataset's size)
TRAIN_SIZE = 10

# 4. Decision Tree settings
CRITERION      = "gini"     # "gini" or "entropy"
MAX_DEPTH      = None       # None = grow full tree; set e.g. 3 to limit depth
MIN_SAMPLES_SPLIT = 2       # minimum samples required to split a node
RANDOM_STATE   = 42

# ============================================================
#  END OF CONFIG — no need to edit below this line
# ============================================================


def encode_dataframe(df):
    """Label-encode all object/string columns. Return encoded df + encoders dict."""
    encoders = {}
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


def gini_node_report(tree_obj):
    """Return a DataFrame with per-node Gini info."""
    rows = []
    for i in range(tree_obj.node_count):
        is_leaf = tree_obj.children_left[i] == -1
        rows.append({
            "Node": i,
            "Gini": round(tree_obj.impurity[i], 4),
            "Samples": tree_obj.n_node_samples[i],
            "Type": "Leaf" if is_leaf else "Internal",
        })
    return pd.DataFrame(rows)


def plot_results(model, X, feature_names, class_names,
                 y_train, train_pred, y_test, test_pred,
                 gini_df, train_acc, test_acc):
    """Draw three panels: decision tree | confusion matrix | gini bar chart."""

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("#0f1117")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.04, right=0.97,
                           top=0.91, bottom=0.07)

    # ── Panel 1: Decision Tree ──────────────────────────────────────────────
    ax_tree = fig.add_subplot(gs[:, 0])
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax_tree,
        impurity=True,
        proportion=False,
    )
    ax_tree.set_title("Decision Tree  (criterion = Gini)",
                      color="white", fontsize=13, pad=10, fontweight="bold")
    ax_tree.set_facecolor("#0f1117")

    # ── Panel 2: Confusion Matrix ───────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(f"Confusion Matrix — Test Set\n"
                    f"Train Acc: {train_acc:.2%}   |   Test Acc: {test_acc:.2%}",
                    color="white", fontsize=11, fontweight="bold")
    ax_cm.set_facecolor("#1a1d27")
    ax_cm.tick_params(colors="white")
    ax_cm.xaxis.label.set_color("white")
    ax_cm.yaxis.label.set_color("white")
    for text in ax_cm.texts:
        text.set_color("black")

    # ── Panel 3: Gini per node ─────────────────────────────────────────────
    ax_gini = fig.add_subplot(gs[1, 1])
    colors = ["#e05c5c" if g > 0.3 else "#5ca8e0" for g in gini_df["Gini"]]
    bars = ax_gini.bar(gini_df["Node"].astype(str), gini_df["Gini"],
                       color=colors, edgecolor="#ffffff22", linewidth=0.5)
    ax_gini.axhline(0.3, color="#f0c040", linewidth=1.2,
                    linestyle="--", label="Gini = 0.30 threshold")
    for bar, val in zip(bars, gini_df["Gini"]):
        if val > 0:
            ax_gini.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.008,
                         f"{val:.3f}",
                         ha="center", va="bottom",
                         color="white", fontsize=7)
    ax_gini.set_title("Gini Index per Node",
                      color="white", fontsize=11, fontweight="bold")
    ax_gini.set_xlabel("Node ID", color="white", fontsize=9)
    ax_gini.set_ylabel("Gini Impurity", color="white", fontsize=9)
    ax_gini.set_ylim(0, 0.65)
    ax_gini.set_facecolor("#1a1d27")
    ax_gini.tick_params(colors="white")
    ax_gini.spines[:].set_color("#ffffff33")
    ax_gini.legend(fontsize=8, labelcolor="white",
                   facecolor="#1a1d27", edgecolor="#ffffff33")

    # ── Super-title ─────────────────────────────────────────────────────────
    fig.suptitle("Gini Decision Tree — Full Analysis",
                 color="white", fontsize=16, fontweight="bold", y=0.97)

    plt.savefig("gini_decision_tree_output.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\n[INFO] Plot saved as  gini_decision_tree_output.png")
    plt.show()


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    # ── 1. Build DataFrame ──────────────────────────────────────────────────
    # ---> CHANGE THIS TO USE AN EXTERNAL DATASET:
    # Comment out the line below and use pd.read_csv() like this:
    # df = pd.read_csv("your_dataset_name.csv")
    df = pd.DataFrame(RAW_DATA)
    
    print("=" * 60)
    print("  Original Dataset")
    print("=" * 60)
    print(df.to_string(index=False))

    # ── 2. Encode ───────────────────────────────────────────────────────────
    df_enc, encoders = encode_dataframe(df)

    # ── 3. Split features / target ──────────────────────────────────────────
    X = df_enc.drop(columns=[TARGET_COLUMN])
    y = df_enc[TARGET_COLUMN]

    if TRAIN_SIZE >= len(df):
        raise ValueError(
            f"TRAIN_SIZE ({TRAIN_SIZE}) must be less than "
            f"total rows ({len(df)})."
        )

    X_train, X_test = X.iloc[:TRAIN_SIZE], X.iloc[TRAIN_SIZE:]
    y_train, y_test = y.iloc[:TRAIN_SIZE], y.iloc[TRAIN_SIZE:]

    # ── 4. Train ────────────────────────────────────────────────────────────
    model = DecisionTreeClassifier(
        criterion=CRITERION,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # ── 5. Accuracy ─────────────────────────────────────────────────────────
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    train_acc  = accuracy_score(y_train, train_pred)
    test_acc   = accuracy_score(y_test,  test_pred)

    print("\n" + "=" * 60)
    print("  Accuracy")
    print("=" * 60)
    print(f"  Training Accuracy : {train_acc:.4f}  ({train_acc:.2%})")
    print(f"  Testing  Accuracy : {test_acc:.4f}  ({test_acc:.2%})")

    # ── 6. Gini per node ────────────────────────────────────────────────────
    gini_df = gini_node_report(model.tree_)
    print("\n" + "=" * 60)
    print("  Gini Index per Node")
    print("=" * 60)
    print(gini_df.to_string(index=False))

    # ── 7. Class names for plots ────────────────────────────────────────────
    if TARGET_COLUMN in encoders:
        class_names = list(encoders[TARGET_COLUMN].classes_)
    else:
        class_names = [str(c) for c in sorted(y.unique())]

    # ── 8. Plot ─────────────────────────────────────────────────────────────
    plot_results(
        model       = model,
        X           = X,
        feature_names = list(X.columns),
        class_names   = class_names,
        y_train     = y_train,
        train_pred  = train_pred,
        y_test      = y_test,
        test_pred   = test_pred,
        gini_df     = gini_df,
        train_acc   = train_acc,
        test_acc    = test_acc,
    )


if __name__ == "__main__":
    main()
