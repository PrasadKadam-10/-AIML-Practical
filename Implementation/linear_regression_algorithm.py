# Practical: Linear Regression
# Objective: Predict housing prices using the California Housing dataset
#            and evaluate model performance with MSE and R² score.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# Step 1: Load dataset
# --------------------------------------------------
housing_data = fetch_california_housing()
X = housing_data.data
y = housing_data.target

print("Dataset shape — Features:", X.shape, "| Target:", y.shape)

# --------------------------------------------------
# Step 2: Split into training and testing sets
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# --------------------------------------------------
# Step 3: Create and train the Linear Regression model
# --------------------------------------------------
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# --------------------------------------------------
# Step 4: Make predictions on the test set
# --------------------------------------------------
y_pred = reg_model.predict(X_test)

# --------------------------------------------------
# Step 5: Evaluate model performance
# --------------------------------------------------
mse_val = mean_squared_error(y_test, y_pred)
r2_val  = r2_score(y_test, y_pred)

print("\nModel Coefficients :", reg_model.coef_)
print("Intercept          :", reg_model.intercept_)
print("Mean Squared Error :", round(mse_val, 4))
print("R² Score           :", round(r2_val, 4))

# --------------------------------------------------
# Step 6: Plot Actual vs Predicted values
# --------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=15)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.title('Linear Regression — Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.tight_layout()
plt.show()
