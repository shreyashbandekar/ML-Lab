import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def kmeans(X, K, max_iters=100):
  centroids = X[:K]

  for _ in range(max_iters):
    # Assign each data point to the nearest centroid

    expanded_x = X[:, np.newaxis]
    euc_dist = np.linalg.norm(expanded_x - centroids, axis=2)
    labels = np.argmin(euc_dist, axis=1)

    # Update the centroids based on the assigned point
    new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

    # If the centroids did not change, stop iterating
    if np.all(centroids == new_centroids):
      break

    centroids = new_centroids

  return labels, centroids


X = load_iris() .data
K=3
labels, centroids = kmeans(X, K)
print("Labels:", labels)
print("Centroids:", centroids)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering of Iris Dataset')
plt.show()

#using sklearn


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
X = load_iris().data

# K-means using scikit-learn
K = 3
kmeans_sklearn = KMeans(n_clusters=K, random_state=0)
labels_sklearn = kmeans_sklearn.fit_predict(X)
centroids_sklearn = kmeans_sklearn.cluster_centers_

print("Scikit-learn K-means Labels:", labels_sklearn)
print("Scikit-learn K-means Centroids:", centroids_sklearn)

# Plotting K-means results using sklearn
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_sklearn)
plt.scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scikit-learn K-means Clustering of Iris Dataset')
plt.show()
