# Practical: SVR with Regularization
# Objective: Fit a Support Vector Regression model with RBF kernel
#            and use regularization (C, epsilon) to control overfitting.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Step 1: Prepare sample dataset
# X = years of experience, y = approximate salary (lakhs)
# --------------------------------------------------
experience = np.array([[1], [2], [3], [4], [5],
                        [6], [7], [8], [9], [10]])
salary     = np.array([2.5, 3.8, 5.0, 6.2, 7.8,
                        9.0, 10.5, 12.1, 13.8, 15.0])

# --------------------------------------------------
# Step 2: Scale features and target (SVR requires scaling)
# --------------------------------------------------
scale_X = StandardScaler()
scale_y = StandardScaler()

X_sc = scale_X.fit_transform(experience)
y_sc = scale_y.fit_transform(salary.reshape(-1, 1))

# --------------------------------------------------
# Step 3: Train SVR model
# C controls regularization: smaller C = more regularization
# epsilon defines the margin of tolerance
# --------------------------------------------------
svr = SVR(kernel='rbf', C=10, epsilon=0.2, gamma=0.1)
svr.fit(X_sc, y_sc.ravel())

# --------------------------------------------------
# Step 4: Predict for a new input value
# --------------------------------------------------
new_exp       = np.array([[5.5]])
new_exp_sc    = scale_X.transform(new_exp)
pred_sc       = svr.predict(new_exp_sc)
predicted_sal = scale_y.inverse_transform(pred_sc.reshape(-1, 1))

print(f"Predicted salary for 5.5 years of experience: {predicted_sal[0][0]:.2f} lakhs")

# --------------------------------------------------
# Step 5: Visualize actual data vs SVR prediction curve
# --------------------------------------------------
fitted = scale_y.inverse_transform(
    svr.predict(X_sc).reshape(-1, 1))

plt.figure(figsize=(7, 5))
plt.scatter(experience, salary, color='crimson', label='Actual Data', zorder=5)
plt.plot(experience, fitted, color='navy', label='SVR Fit (Regularized)')
plt.title('SVR with Regularization — Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (Lakhs)')
plt.legend()
plt.tight_layout()
plt.show()
