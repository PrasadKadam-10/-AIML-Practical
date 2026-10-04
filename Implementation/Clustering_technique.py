# Practical: Clustering Techniques
# Objective: Group students based on their academic performance
#            using K-Means, Hierarchical, and DBSCAN clustering.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# --------------------------------------------------
# Step 1: Create sample dataset
# --------------------------------------------------
records = {
    'Name'   : ['Amit', 'Priya', 'Ravi', 'Sneha', 'Karan',
                 'Meera', 'Arun', 'Divya'],
    'Physics': [82, 74, 91, 68, 62, 85, 53, 57],
    'Chemistry': [78, 71, 88, 70, 58, 83, 48, 55]
}

df = pd.DataFrame(records)
print("Student Marks Dataset:")
print(df)

# Features used for clustering
features = df[['Physics', 'Chemistry']]

# --------------------------------------------------
# K-MEANS CLUSTERING
# --------------------------------------------------
km = KMeans(n_clusters=2, random_state=10, n_init=10)
df['KMeans_Label'] = km.fit_predict(features)

print("\nK-Means Cluster Assignments:")
print(df[['Name', 'KMeans_Label']])

plt.figure(figsize=(6, 4))
plt.scatter(df['Physics'], df['Chemistry'],
            c=df['KMeans_Label'], cmap='viridis', s=100)
for idx, name in enumerate(df['Name']):
    plt.annotate(name, (df['Physics'][idx] + 0.5, df['Chemistry'][idx] + 0.5))
plt.title('K-Means Clustering (Student Marks)')
plt.xlabel('Physics Score')
plt.ylabel('Chemistry Score')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# HIERARCHICAL CLUSTERING
# --------------------------------------------------
link_matrix = linkage(features, method='ward')

plt.figure(figsize=(8, 4))
dendrogram(link_matrix, labels=df['Name'].values)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Student Name')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# DBSCAN CLUSTERING
# --------------------------------------------------
db = DBSCAN(eps=8, min_samples=2)
df['DBSCAN_Label'] = db.fit_predict(features)

print("\nDBSCAN Cluster Assignments:")
print(df[['Name', 'DBSCAN_Label']])

plt.figure(figsize=(6, 4))
plt.scatter(df['Physics'], df['Chemistry'],
            c=df['DBSCAN_Label'], cmap='plasma', s=100)
for idx, name in enumerate(df['Name']):
    plt.annotate(name, (df['Physics'][idx] + 0.5, df['Chemistry'][idx] + 0.5))
plt.title('DBSCAN Clustering (Student Marks)')
plt.xlabel('Physics Score')
plt.ylabel('Chemistry Score')
plt.tight_layout()
plt.show()
