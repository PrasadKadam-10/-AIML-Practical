# Step 0: Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load dataset (example: Iris dataset)
from sklearn.datasets import load_iris
data = load_iris()
X = data.data       # Features
y = data.target     # Labels

# Step 2: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
# Reduce to 2 principal components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Sum of explained variance:", sum(pca.explained_variance_ratio_))

# Step 5: Visualize PCA result
plt.figure(figsize=(8,6))
for i, target_name in enumerate(data.target_names):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label=target_name)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()
