import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt
 

def load_dataset(split):
    """Load and split the dataset into training and test parts.
    
    Parameters
    ----------
    split : float in range (0, 1)
        Fraction of the data used for training.
    
    Returns
    -------
    X_train : array, shape (N_train, 4)
        Training features.
    y_train : array, shape (N_train)
        Training labels.
    X_test : array, shape (N_test, 4)
        Test features.
    y_test : array, shape (N_test)
        Test labels.
    """
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=123, test_size=(1 - split))
    return X_train, X_test, y_train, y_test


# prepare data
# split = 0.75
split = 0.1
X_train, X_test, y_train, y_test = load_dataset(split)


# plot dataset
# Since the data has 4 features, 16 scatterplots (4x4) are plotted showing the dependencies between each pair of features.
f, axes = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        if j == 0 and i == 0:
            axes[i,j].text(0.5, 0.5, 'Sepal. length', ha='center', va='center', size=24, alpha=.5)
        elif j == 1 and i == 1:
            axes[i,j].text(0.5, 0.5, 'Sepal. width', ha='center', va='center', size=24, alpha=.5)
        elif j == 2 and i == 2:
            axes[i,j].text(0.5, 0.5, 'Petal. length', ha='center', va='center', size=24, alpha=.5)
        elif j == 3 and i == 3:
            axes[i,j].text(0.5, 0.5, 'Petal. width', ha='center', va='center', size=24, alpha=.5)
        else:
            axes[i,j].scatter(X_train[:,j], X_train[:,i], c=y_train, cmap=plt.cm.cool)
plt.show()

# 4 upcoming functions needed to perform predictions for new data points
def euclidean_distance(x1, x2):
    """Compute pairwise Euclidean distances between two data points.
    
    Parameters
    ----------
    x1 : array, shape (N, 4)
        First set of data points.
    x2 : array, shape (M, 4)
        Second set of data points.
    
    Returns
    -------
    distance : float array, shape (N, M)
        Pairwise Euclidean distances between x1 and x2.
    """
    N = len(x1[:, 0])
    M = len(x2[:, 0])
    l2dist = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            l2dist[n, m] = np.linalg.norm(x1[n, :] - x2[m, :])

    return l2dist

"""
Efficent Alternative
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1[:, None] - x2[None])**2, -1)) 
"""

def get_neighbors_labels(X_train, y_train, X_new, k):
    """Get the labels of the k nearest neighbors of the datapoints x_new.
    
    Parameters
    ----------
    X_train : array, shape (N_train, 4)
        Training features.
    y_train : array, shape (N_train)
        Training labels.
    X_new : array, shape (M, 4)
        Data points for which the neighbors have to be found.
    k : int
        Number of neighbors to return.
        
    Returns
    -------
    neighbors_labels : array, shape (M, k)
        Array containing the labels of the k nearest neighbors.
    """
    M = len(X_new[:, 0])
    all_distances = euclidean_distance(X_new, X_train)
    k_smallest_labels = np.zeros([M, k])

    for m in range(M):
        mth_slice = all_distances[m, :].copy()
        indices_of_k_smallest = np.zeros(k, dtype=int)

        for i in range(k):
            indices_of_k_smallest[i] = mth_slice.argmin()
            k_smallest_labels[m, i] = y_train[indices_of_k_smallest[i]]
            mth_slice[mth_slice.argmin()] += mth_slice[mth_slice.argmax()]
    
    return k_smallest_labels

"""
Efficient Alternative
def get_neighbors_labels1(X_train, y_train, X_new, k):
    distances = euclidean_distance(X_new, X_train)
    nearest = np.argsort(distances, axis=1)[:, :k]

    return y_train[nearest]
"""


def get_response(neighbors_labels, num_classes=3):
    """Predict label given the set of neighbors.
    
    Parameters
    ----------
    neighbors_labels : array, shape (M, k)
        Array containing the labels of the k nearest neighbors per data point.
    num_classes : int
        Number of classes in the dataset.
    
    Returns
    -------
    y : int array, shape (M,)
        Majority class among the neighbors.
        In case of tie, return smallest class value.
    """
    M = len(neighbors_labels[:, 0])
    majority_class_array = np.zeros(M, dtype=int)
    maj_class = 9
    for m in range(M):
        class0_count = 0
        class1_count = 0
        class2_count = 0
        for label in neighbors_labels[m, :]:
            if label == 0:
                class0_count += 1
            if label == 1:
                class1_count += 1
            if label == 2:
                class2_count += 1
        if class0_count == class1_count == class2_count:
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

    return majority_class_array

"""
Efficient Alternative
def get_response1(neighbors_labels, num_classes=3):
    class_votes = (neighbors_labels[:, :, None] == np.arange(num_classes)[None, None]).sum(1)
    print(np.argmax(class_votes, 1))
    return np.argmax(class_votes, 1)
"""

def compute_accuracy(y_pred, y_test):
    """Compute accuracy of prediction.
    
    Parameters
    ----------
    y_pred : array, shape (N_test)
        Predicted labels.
    y_test : array, shape (N_test)
        True labels.
    """
    length_test_set = len(y_test)
    counter = 0
    for i in range(length_test_set):
        if y_pred[i] != y_test[i]:
            counter += 1
    inaccuracy = counter/length_test_set
    accuracy = 1 - inaccuracy

    return accuracy

"""
Efficient Alternative 
    Uses boolean values while previous function uses an inaccuracy counter
def compute_accuracy(y_pred, y_test):
    return np.mean(y_pred == y_test)
"""

# Now we want to find the labels of the k-nearest neighbors to our test set of datapoints 
# We choose k = 3
k = 3
neighbors = get_neighbors_labels(X_train, y_train, X_test, k)

# Next we want to predict what the most likely labels will be for this test set of datapoints
y_pred = get_response(neighbors)

# prepare data
#split = 0.75
#X_train, X_test, y_train, y_test = load_dataset(split)
print('Training set: {0} samples'.format(X_train.shape[0]))
print('Test set: {0} samples'.format(X_test.shape[0]))

# generate predictions
accuracy = compute_accuracy(y_pred, y_test)
print('Accuracy = {0}'.format(accuracy))