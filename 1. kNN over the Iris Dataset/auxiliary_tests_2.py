# These are tests I used to help me program certain functions
# This file is not meant to be read or interpreted by outside parties

import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
    N = len(x1[:, 0])
    M = len(x2[:, 0])
    l2dist = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            l2dist[n, m] = np.linalg.norm(x1[n, :] - x2[m, :])
    return l2dist

k = 3
X_train = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])
y_train = np.array([1, 2, 3, 4, 5])
X_new = np.array([[10, 20, 30, 40], [4, 1, 3, 2]])

N_train = len(X_train[:, 0])
M = len(X_new[:, 0])
all_distances = euclidean_distance(X_new, X_train)
print(all_distances)
k_smallest_labels = np.zeros([M, k])

for m in range(M):
    mth_slice = all_distances[m, :].copy()
    indices_of_k_smallest = np.zeros(k, dtype=int)

    for i in range(k):
        indices_of_k_smallest[i] = mth_slice.argmin()
        k_smallest_labels[m, i] = y_train[indices_of_k_smallest[i]]
        mth_slice[mth_slice.argmin()] += mth_slice[mth_slice.argmax()]

print(k_smallest_labels)

## Efficient Alternative ##
distances = euclidean_distance(X_new, X_train)
nearest = np.argsort(distances, axis=1)[:, :k]
print(y_train[nearest])
