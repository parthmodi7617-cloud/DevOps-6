import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

# Load dataset
# [CHANGE HERE] If using your own dataset (like a CSV), load it directly using pandas:
# data = pd.read_csv("your_dataset.csv")
# You can then remove the fetch_california_housing() and DataFrame creation parts below.
housing = fetch_california_housing()

# Create DataFrame
data = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Add target column
data['Price'] = housing.target

# Features and target
# [CHANGE HERE] Replace 'Price' with the actual name of the column you want to predict in your new dataset.
X = data.drop('Price', axis=1)
y = data['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Sample prediction
print("\nSample Prediction:")
# [CHANGE HERE] Update the print labels below to match what your new dataset represents
print("Predicted Price:", y_pred[0])
print("Actual Price   :", y_test.iloc[0])

# [CHANGE HERE] If you want to predict on a completely new, manually entered data point:
# Note: The values list must match the number and order of features in your dataset.
# completely_new_data = pd.DataFrame([[8.3, 41.0, 6.9, 1.0, 322.0, 2.5, 37.8, -122.2]], columns=X.columns)
# new_pred = model.predict(completely_new_data)
# print("Prediction for new input:", new_pred[0])

# Plot
plt.scatter(y_test, y_pred)

# [CHANGE HERE] Update the axis labels below to match your new target variable
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")

plt.show()