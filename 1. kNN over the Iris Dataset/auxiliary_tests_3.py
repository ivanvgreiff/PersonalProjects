# These are tests I used to help me program certain functions
# This file is not meant to be read or interpreted by outside parties

import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt


neighbor_labels = np.array([[1, 2, 1, 1], [2, 0, 2, 0], [1, 2, 0, 0], [1, 2, 2, 2], [0, 2, 1, 1]])
answer = np.array([1, 0, 0, 2, 1])

M = len(neighbor_labels[:, 0])
majority_class_array = np.zeros(M, dtype=int)
maj_class = 9
for m in range(M):
    class0_count = 0
    class1_count = 0
    class2_count = 0
    for label in neighbor_labels[m, :]:
        if label == 0:
            class0_count += 1
        if label == 1:
            class1_count += 1
        if label == 2:
            class2_count += 1
    if class0_count == class1_count == class2_count: ## previously used 'and' instead of '==' for second comparator, caused issues
        maj_class = 0
    elif class0_count == class1_count:
        if class0_count < class2_count:
            maj_class = 2
        else:
            maj_class = 0
    elif class1_count == class2_count:
        if class0_count < class1_count:
            maj_class = 1
        else:
            maj_class = 0
    elif class0_count == class2_count:
        if class0_count < class1_count:
            maj_class = 1
        else:
            maj_class = 0
    else:
        maj_class = np.array([class0_count, class1_count, class2_count]).argmax()
    majority_class_array[m] = maj_class
print(majority_class_array)


## Efficient Alternative ##
num_classes = 3
class_votes = (neighbor_labels[:, :, None] == np.arange(num_classes)[None, None]).sum(1)
print(np.argmax(class_votes, 1))