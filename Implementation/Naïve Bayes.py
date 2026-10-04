# Practical: Naive Bayes Classifier
# Objective: Classify wine samples into categories using the
#            Gaussian Naive Bayes algorithm and evaluate performance.

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# Step 1: Load the Wine dataset
# --------------------------------------------------
wine = load_wine()
X = wine.data    # 13 chemical feature columns
y = wine.target  # 3 wine classes

print("Feature names:", wine.feature_names)
print("Classes:", wine.target_names)
print("Dataset shape:", X.shape)

# --------------------------------------------------
# Step 2: Split data into training and test sets
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1)

# --------------------------------------------------
# Step 3: Standardize features
# Scaling helps Gaussian NB model distributions better.
# --------------------------------------------------
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)

# --------------------------------------------------
# Step 4: Train the Gaussian Naive Bayes classifier
# --------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train_sc, y_train)

# --------------------------------------------------
# Step 5: Predict on test data
# --------------------------------------------------
y_pred = gnb.predict(X_test_sc)

# --------------------------------------------------
# Step 6: Evaluate the model
# --------------------------------------------------
acc     = accuracy_score(y_test, y_pred)
cm      = confusion_matrix(y_test, y_pred)
report  = classification_report(y_test, y_pred,
                                target_names=wine.target_names)

print("\nAccuracy        :", round(acc, 4))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
