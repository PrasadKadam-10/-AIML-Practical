# SVR with Regularization Example

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Step 1: Prepare Dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([1.2, 1.9, 3.2, 3.8, 5.1, 5.8, 7.2, 7.9, 9.1, 10.2])

# Step 2: Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1))

# Step 3: Train SVR Model with Regularization
# Set smaller C and moderate epsilon to avoid overfitting
svr_regressor = SVR(kernel='rbf', C=10, epsilon=0.2, gamma=0.1)
svr_regressor.fit(X_scaled, y_scaled.ravel())

# Step 4: Prediction
X_test = np.array([[6.5]])
X_test_scaled = sc_X.transform(X_test)
y_pred_scaled = svr_regressor.predict(X_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

print("Predicted value for X=6.5:", y_pred[0][0])

# Step 5: Visualize Results
plt.scatter(X, y, color='red', label='Original Data')
plt.plot(X, sc_y.inverse_transform(svr_regressor.predict(X_scaled).reshape(-1, 1)),
         color='blue', label='SVR Prediction (Regularized)')
plt.title('SVR Regression with Regularization')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
