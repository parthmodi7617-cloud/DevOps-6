import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Dataset
# TODO 1: Load your custom dataset.
# Example: data = pd.read_csv('your_dataset.csv')
data = pd.DataFrame({
    "Age": [22,25,28,30,35,40,42,45,50,55,
            23,27,33,38,48,29,36,41,52,24],

    "Annual_Income": [25000,27000,30000,32000,40000,
                      45000,48000,50000,52000,60000,
                      26000,29000,38000,43000,51000,
                      31000,41000,47000,55000,25500],

    "Spending_Score": [30,35,40,42,60,65,70,75,80,85,
                       32,38,55,62,78,41,58,68,82,34],

    "Purchase": ["No","No","No","No","Yes",
                 "Yes","Yes","Yes","Yes","Yes",
                 "No","No","Yes","Yes","Yes",
                 "No","Yes","Yes","Yes","No"]
})

# Features and target
# TODO 2: Update the column names below to match the features and target labels in your dataset
X = data[["Age", "Annual_Income", "Spending_Score"]] # Example: data[["Feature1", "Feature2"]]
y = data["Purchase"] # Example: data["Target_Column"]

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Test different K values
k_values = range(1, 11)
accuracies = []

print("KNN Results\n")

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print("K =", k)
    print("Accuracy :", accuracy)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

# Best K
best_k = k_values[accuracies.index(max(accuracies))]

print("Best K value:", best_k)

# Final model
final_model = KNeighborsClassifier(
    n_neighbors=best_k,
    metric='euclidean'
)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

# Plot Accuracy vs K
plt.plot(k_values, accuracies, marker='o')

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")

plt.grid(True)
plt.show()

# Scatter Plot
# TODO 3: Update the columns below to choose which two features to visualize from your new dataset
plt.scatter(
    data["Annual_Income"], # Replace with your dataset's feature for X-axis
    data["Spending_Score"], # Replace with your dataset's feature for Y-axis
    c=y
)

# TODO 4: Update the axis labels and title to describe your custom dataset properties
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Data")

plt.show()