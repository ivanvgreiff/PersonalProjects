# These are tests I used to help me program certain functions
# This file is not meant to be read or interpreted by outside parties

import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt

x1 = np.array([[1, 8, 2], [2, 0, 2]])
x2 = np.array([[1, 8, 2], [1, 1, 1], [1, 1, 1]])

N = len(x1[:, 0])
M = len(x2[:, 0])
l2dist = np.zeros((N, M))
for n in range(N):
    for m in range(M):
        l2dist[n, m] = np.linalg.norm(x1[n, :] - x2[m, :])

print(l2dist)

print(np.sqrt(np.sum((x1[:, None] - x2[None])**2, -1)))
print(x1[:, None])
print(x2[None])
print(x1[:, None] - x2[None])

a = np.array([1, 2, 3])
print(a)
print(a[None])
print(a[:, None])
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
print(b[None])
print(b[:, None])

print('HERE')
x = np.array([[1, 2], [4, 5]])
y = np.array([[6, 6]])
print(x - y)