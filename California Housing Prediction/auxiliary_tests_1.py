# These are tests I used to help me program certain functions
# This file is not meant to be read or interpreted by outside parties

import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1, 0, 0, 0])

dimY = len(y_true)
mse = 1/dimY * ((y_true - y_pred)**2).sum()

print(mse)

print(5**(1/3))