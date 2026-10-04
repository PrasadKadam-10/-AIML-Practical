# Practical: Dimensionality Reduction using PCA
# Objective: Reduce the number of features in the Iris dataset
#            from 4 to 2 principal components and visualize the result.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# Step 1: Load the Iris dataset
# --------------------------------------------------
iris = load_iris()
X = iris.data      # 4 feature columns
y = iris.target    # class labels (0, 1, 2)

print("Original feature shape:", X.shape)

# --------------------------------------------------
# Step 2: Standardize the features
# PCA is sensitive to scale, so we normalize first.
# --------------------------------------------------
normalizer = StandardScaler()
X_norm = normalizer.fit_transform(X)

# --------------------------------------------------
# Step 3: Apply PCA — reduce to 2 components
# --------------------------------------------------
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_norm)

print("Reduced feature shape:", X_reduced.shape)
print("Explained variance per component:", pca.explained_variance_ratio_)
print("Total variance retained: {:.2f}%".format(
    sum(pca.explained_variance_ratio_) * 100))

# --------------------------------------------------
# Step 4: Plot the 2D projection
# --------------------------------------------------
colors = ['darkorange', 'steelblue', 'green']
plt.figure(figsize=(8, 6))

for class_id, class_name in enumerate(iris.target_names):
    mask = y == class_id
    plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                label=class_name, color=colors[class_id], s=60)

plt.title('PCA — Iris Dataset (2 Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.tight_layout()
plt.show()
