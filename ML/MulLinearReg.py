import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
# --- CHANGE 1: Load your actual dataset here ---
# Example: data = pd.read_csv("path/to/your/dataset.csv")
data = pd.DataFrame({
    "Experience": [1,2,3,4,5,6,7,8,9,10],
    "Hours_Studied": [2,3,4,5,6,7,8,9,10,11],
    "Salary": [25000,30000,35000,40000,45000,
               50000,55000,60000,65000,70000]
})

# Features and target
# --- CHANGE 2: Specify the columns you want to use as features (X) ---
X = data[["Experience", "Hours_Studied"]]

# --- CHANGE 3: Specify the column you want to predict (target, y) ---
y = data["Salary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Print coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Print predictions
print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Plot
plt.scatter(y_test, y_pred)

# --- CHANGE 4: Update the plot labels to match your new target variable ---
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted")

# Perfect prediction line
plt.plot([y.min(), y.max()],
         [y.min(), y.max()])

plt.show()