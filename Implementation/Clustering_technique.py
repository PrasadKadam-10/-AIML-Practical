# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 2: Create Sample Data
data = {
    'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Hannah'],
    'Math': [85, 78, 92, 70, 65, 88, 55, 60],
    'Science': [80, 75, 90, 72, 60, 85, 50, 58]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# Step 3: Select Features for Clustering
X = df[['Math', 'Science']]

# ========================
# K-MEANS CLUSTERING
# ========================
kmeans = KMeans(n_clusters=2, random_state=0)
df['KMeans_Cluster'] = kmeans.fit_predict(X)

print("\nK-Means Clusters:")
print(df[['Student', 'KMeans_Cluster']])

# Plot K-Means Clusters
plt.figure(figsize=(6,4))
plt.scatter(df['Math'], df['Science'], c=df['KMeans_Cluster'], cmap='viridis', s=100)
plt.title('K-Means Clustering of Students')
plt.xlabel('Math Score')
plt.ylabel('Science Score')
for i, txt in enumerate(df['Student']):
    plt.annotate(txt, (df['Math'][i]+0.5, df['Science'][i]+0.5))
plt.show()

# ========================
# HIERARCHICAL CLUSTERING
# ========================
linked = linkage(X, method='ward')
plt.figure(figsize=(8,4))
dendrogram(linked, labels=df['Student'].values)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Students')
plt.ylabel('Distance')
plt.show()

# ========================
# DBSCAN CLUSTERING
# ========================
dbscan = DBSCAN(eps=7, min_samples=2)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X)

print("\nDBSCAN Clusters:")
print(df[['Student', 'DBSCAN_Cluster']])

# Plot DBSCAN Clusters
plt.figure(figsize=(6,4))
plt.scatter(df['Math'], df['Science'], c=df['DBSCAN_Cluster'], cmap='plasma', s=100)
plt.title('DBSCAN Clustering of Students')
plt.xlabel('Math Score')
plt.ylabel('Science Score')
for i, txt in enumerate(df['Student']):
    plt.annotate(txt, (df['Math'][i]+0.5, df['Science'][i]+0.5))
plt.show()
