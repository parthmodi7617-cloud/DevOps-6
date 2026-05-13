import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Titanic Dataset
# CHANGE 1: Replace this hardcoded dataframe with your own dataset, e.g.:
# data = pd.read_csv('path_to_your_dataset.csv')
data = pd.DataFrame({
    'Pclass': [1,3,2,1,3,2,1,3,2,1],
    'Sex': ['male','female','female','male','male',
            'female','female','male','female','male'],
    'Age': [22,38,26,35,28,30,19,40,29,50],
    'Fare': [7.25,71.28,7.92,53.1,8.05,
             13.0,80.0,7.75,26.0,60.0],
    'Survived': [0,1,1,1,0,1,1,0,1,0]
})

# Convert categorical to numeric
# CHANGE 2: Machine learning models require all input data to be numeric.
# If your new dataset contains categorical columns (e.g., text labels like 'red', 'green', 'blue' or 'yes', 'no'),
# you MUST convert them into numbers before training.
#
# Option A: Manual Mapping (Good for simply binary or ordinal categories)
# Example: data['YourColumn'] = data['YourColumn'].map({'yes': 1, 'no': 0})

data['Sex'] = data['Sex'].map({
    'male': 0,
    'female': 1
})

# Features and target
# CHANGE 3: Update the list of columns to match the features (X) and target (y) in your dataset
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Sample prediction
# CHANGE 4: Update this sample to have the same columns and appropriate valid data as your new features (X)
sample = pd.DataFrame({
    'Pclass': [2],
    'Sex': [1],
    'Age': [25],
    'Fare': [30]
})

prediction = model.predict(sample)

print("\nSample Prediction:")
# CHANGE 5: Update the output labels to reflect your new target variable's classes
print("Survived" if prediction[0] == 1
      else "Not Survived")

# Feature Importance Plot
importance = model.feature_importances_

plt.bar(X.columns, importance)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")

plt.show()