from statistics import median
import numpy as np 
from numpy.linalg import norm
from sklearn import datasets
from sklearn.metrics.pairwise import manhattan_distances, pairwise_distances_argmin
RSEED = 2

#dataset
X, y = datasets.load_iris(return_X_y=True)
X = X[:, 0:2]
num_clusters = 3

def kmeans(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters] 
    centers = X[i]

    while True:
     labels = pairwise_distances_argmin(X, centers, metric="manhattan")

     
     new_centers = []
     for i in range(n_clusters):
      updated_center = np.median(X[labels == i], axis=0)
      new_centers.append(updated_center)
            
     new_centers = np.array(new_centers)

     if np.all(centers == new_centers): # Test whether all array elements evaluate to True.
      break
     centers = new_centers
        
    
    return centers, labels


centers, labels = kmeans(X, 4)
print(labels)