# These are tests I used to help me program certain functions
# This file is not meant to be read or interpreted by outside parties

import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt

y_test = np.array([0, 2, 0, 0, 1, 1, 0, 0, 0, 1, 2, 1])
y_pred = np.array([0, 0, 0, 0, 1, 2, 0, 2, 0, 1, 0, 1])

length_test_set = len(y_test)
counter = 0
for i in range(length_test_set):
    if y_pred[i] != y_test[i]:
        counter += 1
inaccuracy = counter/length_test_set
accuracy = 1 - inaccuracy
print(accuracy)
print(y_pred == y_test)
print(np.mean(y_pred == y_test))