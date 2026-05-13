
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Boosting models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb
import lightgbm as lgb


# ============================================================
# Load Iris Dataset
# ============================================================

iris = load_iris()

# Features:
# sepal length, sepal width,
# petal length, petal width

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)


# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# AdaBoost
# ============================================================

ada_model = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)

ada_acc = accuracy_score(y_test, y_pred_ada)

print("AdaBoost Accuracy:", ada_acc)


# ============================================================
# Gradient Boosting
# ============================================================

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)

gb_acc = accuracy_score(y_test, y_pred_gb)

print("Gradient Boosting Accuracy:", gb_acc)


# ============================================================
# XGBoost
# ============================================================

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, y_pred_xgb)

print("XGBoost Accuracy:", xgb_acc)


# ============================================================
# LightGBM
# ============================================================

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)

lgb_acc = accuracy_score(y_test, y_pred_lgb)

print("LightGBM Accuracy:", lgb_acc)


# ============================================================
# Results Table
# ============================================================

results = pd.DataFrame({
    "Model": [
        "AdaBoost",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM"
    ],
    "Accuracy": [
        ada_acc,
        gb_acc,
        xgb_acc,
        lgb_acc
    ]
})

print("\nModel Comparison:")
print(results)