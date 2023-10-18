import numpy as np
from main import *

w = np.array([1, 1, 1])
X = np.array([[2, 2, 2], [3, 3, 3]])

print(X[0, :])
print(w)
print(w@X[0, :].T)